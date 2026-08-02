from __future__ import annotations

import typing

from PySide6 import QtCore, QtGui
from sqlalchemy.orm import make_transient

from splaat.db import PhoneInterval, Utterance, WordInterval

if typing.TYPE_CHECKING:
    from splaat.desktop.models import FileModel


class FileCommand(QtGui.QUndoCommand):
    def __init__(self, file_model: FileModel):
        super().__init__()
        self.file_model = file_model
        self.corpus_model = file_model.corpus_model
        self.resets_tier = False

    def _redo(self, session) -> None:
        pass

    def _undo(self, session) -> None:
        pass

    def update_data(self):
        if self.resets_tier:
            self.file_model.refreshTiers.emit()

    def redo(self) -> None:
        with self.corpus_model.edit_lock:
            try:
                self._redo(self.corpus_model.session)
                self.corpus_model.session.commit()
            except Exception:
                self.corpus_model.session.rollback()
                raise
            # while True:
            #    try:
            #        with self.corpus_model.session.begin_nested():
            #            self._redo()
            #        break
            #    except psycopg2.errors.DeadlockDetected:
            #        pass

        self.update_data()

    def undo(self) -> None:
        with self.corpus_model.edit_lock:
            try:
                self._undo(self.corpus_model.session)
                self.corpus_model.session.commit()
            except Exception:
                self.corpus_model.session.rollback()
                raise
            # while True:
            #    try:
            #        with self.corpus_model.session.begin_nested():
            #            self._undo()
            #        break
            #    except psycopg2.errors.DeadlockDetected:
            #        pass
        self.update_data()


class UpdatePhoneBoundariesCommand(FileCommand):
    def __init__(
        self,
        utterance: Utterance,
        first_phone_interval: PhoneInterval,
        second_phone_interval: PhoneInterval,
        new_time: float,
        file_model: FileModel,
    ):
        super().__init__(file_model)
        self.utterance = utterance
        self.first_phone_interval = first_phone_interval
        self.second_phone_interval = second_phone_interval
        self.first_word_interval = None
        self.second_word_interval = None
        self.at_word_boundary = (
            self.first_phone_interval.word_interval_id
            != self.second_phone_interval.word_interval_id
        )
        if self.at_word_boundary:
            word_intervals = utterance.word_intervals
            for wi in word_intervals:
                if self.first_word_interval is not None and self.second_word_interval is not None:
                    break
                if wi.id == first_phone_interval.word_interval_id:
                    self.first_word_interval = wi
                elif wi.id == second_phone_interval.word_interval_id:
                    self.second_word_interval = wi
        self.old_time = second_phone_interval.start
        self.new_time = new_time
        self.setText(
            QtCore.QCoreApplication.translate(
                "UpdatePhoneBoundariesCommand", "Update phone boundaries"
            )
        )

    def _set_time(self, session, new_time):
        self.first_phone_interval.end = new_time
        self.second_phone_interval.start = new_time
        session.merge(self.utterance)
        session.merge(self.first_phone_interval)
        session.merge(self.second_phone_interval)
        if self.at_word_boundary:
            if self.first_word_interval is not None:
                self.first_word_interval.end = new_time
                session.merge(self.first_word_interval)
            if self.second_word_interval is not None:
                self.second_word_interval.start = new_time
                session.merge(self.second_word_interval)

    def _redo(self, session) -> None:
        self._set_time(session, self.new_time)

    def _undo(self, session) -> None:
        self._set_time(session, self.old_time)

    def id(self) -> int:
        return 2

    def mergeWith(self, other: UpdatePhoneBoundariesCommand) -> bool:
        if (
            other.id() != self.id()
            or other.first_phone_interval.id != self.first_phone_interval.id
            or other.second_phone_interval.id != self.second_phone_interval.id
        ):
            return False
        self.new_time = other.new_time
        return True

    def update_data(self):
        super().update_data()
        self.file_model.changeCommandFired.emit()


class DeletePhoneIntervalCommand(FileCommand):
    def __init__(
        self,
        utterance: Utterance,
        phone_interval: PhoneInterval,
        previous_phone_interval: typing.Optional[PhoneInterval],
        following_phone_interval: typing.Optional[PhoneInterval],
        time_point: typing.Optional[float],
        file_model: FileModel,
    ):
        super().__init__(file_model)
        self.word_interval_lookup = "word_intervals"
        self.phone_interval_lookup = "phone_intervals"
        self.using_reference = not isinstance(phone_interval, PhoneInterval)
        if self.using_reference:
            self.word_interval_lookup = "reference_word_intervals"
            self.phone_interval_lookup = "reference_phone_intervals"
        self.utterance = utterance
        self.phone_interval = phone_interval
        self.previous_phone_interval = previous_phone_interval
        self.has_previous = previous_phone_interval is not None
        self.has_following = following_phone_interval is not None
        self.following_phone_interval = following_phone_interval
        self.new_time = time_point
        self.first_word_interval = None
        self.second_word_interval = None
        previous_word_interval_id = (
            self.previous_phone_interval.word_interval_id
            if self.previous_phone_interval is not None
            else None
        )
        word_interval_id = self.phone_interval.word_interval_id
        following_word_interval_id = (
            self.following_phone_interval.word_interval_id
            if self.following_phone_interval is not None
            else None
        )
        self.word_interval = None
        for wi in getattr(self.utterance, self.word_interval_lookup):
            if wi.id == self.phone_interval.word_interval_id:
                self.word_interval = wi
                break
        self.single_phone_word = (
            word_interval_id != previous_word_interval_id
            and word_interval_id != following_word_interval_id
        )

        self.at_word_boundary = previous_word_interval_id != following_word_interval_id
        if self.at_word_boundary:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                if self.first_word_interval is not None and self.second_word_interval is not None:
                    break
                if wi.id == previous_word_interval_id:
                    self.first_word_interval = wi
                if wi.id == following_word_interval_id:
                    self.second_word_interval = wi
        elif not self.has_previous:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                if wi.id == following_word_interval_id:
                    self.second_word_interval = wi
                    break
        elif not self.has_following:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                if wi.id == previous_word_interval_id:
                    self.first_word_interval = wi
                    break
        self.first_word_interval_end = None
        if self.first_word_interval is not None:
            self.first_word_interval_end = self.first_word_interval.end
        self.second_word_interval_begin = None
        if self.second_word_interval is not None:
            self.second_word_interval_begin = self.second_word_interval.start
        self.setText(
            QtCore.QCoreApplication.translate(
                "DeletePhoneIntervalCommand", "Delete phone interval"
            )
        )

    def _redo(self, session) -> None:
        if self.has_previous and self.has_following:
            new_time = (
                self.new_time
                if self.new_time is not None
                else (self.phone_interval.start + self.phone_interval.end) / 2
            )
            self.previous_phone_interval.end = new_time
            self.following_phone_interval.start = new_time
            if self.at_word_boundary:
                self.first_word_interval.end = new_time
                self.second_word_interval.start = new_time
                session.merge(self.first_word_interval)
                session.merge(self.second_word_interval)
        elif self.has_following:
            self.following_phone_interval.start = self.phone_interval.start
            self.second_word_interval.start = self.phone_interval.start
            session.merge(self.second_word_interval)
        elif self.has_previous:
            self.previous_phone_interval.end = self.phone_interval.end
            self.first_word_interval.end = self.phone_interval.end
            session.merge(self.first_word_interval)
        if self.has_previous:
            session.merge(self.previous_phone_interval)
        if self.has_following:
            session.merge(self.following_phone_interval)

        phone_intervals = []
        for pi in getattr(self.utterance, self.phone_interval_lookup):
            if pi.id == self.phone_interval.id:
                continue
            session.merge(pi)
            phone_intervals.append(pi)
        setattr(self.utterance, self.phone_interval_lookup, phone_intervals)
        if self.single_phone_word:
            word_intervals = []
            for wi in getattr(self.utterance, self.word_interval_lookup):
                if wi.id == self.word_interval.id:
                    continue
                session.merge(wi)
                word_intervals.append(wi)
            setattr(self.utterance, self.word_interval_lookup, word_intervals)
        session.merge(self.utterance)

    def _undo(self, session) -> None:
        if self.single_phone_word:
            word_intervals = []
            for wi in getattr(self.utterance, self.word_interval_lookup):
                session.merge(wi)
                word_intervals.append(wi)
            make_transient(self.word_interval)
            word_intervals.append(self.word_interval)
            setattr(
                self.utterance,
                self.word_interval_lookup,
                sorted(word_intervals, key=lambda x: x.start),
            )
        phone_intervals = []
        for pi in getattr(self.utterance, self.phone_interval_lookup):
            session.merge(pi)
            phone_intervals.append(pi)
        make_transient(self.phone_interval)
        phone_intervals.append(self.phone_interval)
        setattr(
            self.utterance,
            self.phone_interval_lookup,
            sorted(phone_intervals, key=lambda x: x.start),
        )
        session.merge(self.utterance)

        if self.has_previous:
            self.previous_phone_interval.end = self.phone_interval.start
            session.merge(self.previous_phone_interval)
        if self.has_following:
            self.following_phone_interval.start = self.phone_interval.end
            session.merge(self.following_phone_interval)
        if self.second_word_interval_begin is not None:
            self.second_word_interval.start = self.second_word_interval_begin
            session.merge(self.second_word_interval)
        if self.first_word_interval_end is not None:
            self.first_word_interval.end = self.first_word_interval_end
            session.merge(self.first_word_interval)

    def update_data(self):
        super().update_data()
        self.file_model.changeCommandFired.emit()
        self.file_model.phoneTierChanged.emit(self.utterance)


class InsertPhoneIntervalCommand(FileCommand):
    def __init__(
        self,
        utterance: Utterance,
        phone_interval: PhoneInterval,
        previous_phone_interval: typing.Optional[PhoneInterval],
        following_phone_interval: typing.Optional[PhoneInterval],
        file_model: FileModel,
        word_interval: WordInterval = None,
    ):
        super().__init__(file_model)
        self.word_interval_lookup = "word_intervals"
        self.phone_interval_lookup = "phone_intervals"
        self.utterance = utterance
        self.phone_interval = phone_interval
        self.previous_phone_interval = previous_phone_interval
        self.has_previous = previous_phone_interval is not None
        self.has_following = following_phone_interval is not None
        self.following_phone_interval = following_phone_interval
        self.word_interval = word_interval
        self.previous_word_interval_id = (
            previous_phone_interval.word_interval_id if self.has_previous else None
        )
        self.following_word_interval_id = (
            following_phone_interval.word_interval_id if self.has_following else None
        )
        self.initial_word_boundary = (
            self.has_previous
            and self.previous_word_interval_id != self.phone_interval.word_interval_id
        )
        self.final_word_boundary = (
            self.has_following
            and self.following_word_interval_id != self.phone_interval.word_interval_id
        )

        self.old_time_boundary = (
            self.previous_phone_interval.end
            if self.has_previous
            else self.following_phone_interval.start
        )
        self.previous_word_interval_end = None
        self.following_word_interval_begin = None
        for wi in getattr(self.utterance, self.word_interval_lookup):
            if (
                self.previous_word_interval_id is not None
                and wi.id == self.previous_word_interval_id
            ):
                self.previous_word_interval_end = wi.end
            elif (
                self.following_word_interval_id is not None
                and wi.id == self.following_word_interval_id
            ):
                self.following_word_interval_begin = wi.start
        self.setText(
            QtCore.QCoreApplication.translate(
                "InsertPhoneIntervalCommand", "Insert phone interval"
            )
        )

    def _redo(self, session) -> None:
        word_intervals = []
        if self.word_interval is not None:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                session.merge(wi)
                if wi.id == self.previous_word_interval_id:
                    wi.end = self.phone_interval.start
                if wi.id == self.following_word_interval_id:
                    wi.start = self.phone_interval.end
                word_intervals.append(wi)
            make_transient(self.word_interval)
            word_intervals.append(self.word_interval)
        else:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                session.merge(wi)
                if self.initial_word_boundary:
                    if wi.id == self.previous_word_interval_id:
                        wi.end = self.phone_interval.start
                    elif wi.id == self.phone_interval.word_interval_id:
                        wi.start = self.phone_interval.start
                if self.final_word_boundary:
                    if wi.id == self.following_word_interval_id:
                        wi.start = self.phone_interval.end
                    elif wi.id == self.phone_interval.word_interval_id:
                        wi.end = self.phone_interval.end
                word_intervals.append(wi)
        setattr(
            self.utterance,
            self.word_interval_lookup,
            sorted(word_intervals, key=lambda x: x.start),
        )
        phone_intervals = []
        for pi in getattr(self.utterance, self.phone_interval_lookup):
            session.merge(pi)
            if self.has_previous and pi.id == self.previous_phone_interval.id:
                pi.end = self.phone_interval.start
            if self.has_following and pi.id == self.following_phone_interval.id:
                pi.start = self.phone_interval.end
            phone_intervals.append(pi)
        make_transient(self.phone_interval)
        self.phone_interval.utterance_id = self.utterance.id
        phone_intervals.append(self.phone_interval)

        setattr(
            self.utterance,
            self.phone_interval_lookup,
            sorted(phone_intervals, key=lambda x: x.start),
        )
        session.merge(self.utterance)

    def _undo(self, session) -> None:
        phone_intervals = []
        for pi in getattr(self.utterance, self.phone_interval_lookup):
            if pi.id == self.phone_interval.id:
                continue
            session.merge(pi)
            if self.has_previous and pi.id == self.previous_phone_interval.id:
                pi.end = self.old_time_boundary
            if self.has_following and pi.id == self.following_phone_interval.id:
                pi.start = self.old_time_boundary
            phone_intervals.append(pi)
        setattr(self.utterance, self.phone_interval_lookup, phone_intervals)
        word_intervals = []
        if self.word_interval is not None:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                if wi.id == self.word_interval.id:
                    continue
                session.merge(wi)
                word_intervals.append(wi)
        else:
            for wi in getattr(self.utterance, self.word_interval_lookup):
                session.merge(wi)
                if self.initial_word_boundary:
                    if wi.id == self.previous_word_interval_id:
                        wi.end = self.previous_word_interval_end
                    elif wi.id == self.phone_interval.word_interval_id:
                        wi.start = self.previous_word_interval_end
                if self.final_word_boundary:
                    if wi.id == self.following_word_interval_id:
                        wi.start = self.following_word_interval_begin
                    elif wi.id == self.phone_interval.word_interval_id:
                        wi.end = self.following_word_interval_begin
                word_intervals.append(wi)
        setattr(self.utterance, self.word_interval_lookup, word_intervals)
        session.merge(self.utterance)

    def update_data(self):
        super().update_data()
        self.file_model.changeCommandFired.emit()
        self.file_model.phoneTierChanged.emit(self.utterance)


class UpdatePhoneIntervalCommand(FileCommand):
    def __init__(
        self,
        utterance: Utterance,
        phone_interval: PhoneInterval,
        new_phone: str,
        file_model: FileModel,
    ):
        super().__init__(file_model)
        self.utterance = utterance
        self.phone_interval = phone_interval
        self.old_phone = self.phone_interval.phone
        self.new_phone = new_phone
        self.setText(
            QtCore.QCoreApplication.translate(
                "UpdatePhoneIntervalCommand", "Update phone interval"
            )
        )

    def _redo(self, session) -> None:
        self.phone_interval.phone = self.new_phone
        session.merge(self.phone_interval)

    def _undo(self, session) -> None:
        self.phone_interval.phone = self.old_phone
        session.merge(self.phone_interval)

    def update_data(self):
        super().update_data()
        self.file_model.changeCommandFired.emit()
        self.file_model.phoneTierChanged.emit(self.utterance)
