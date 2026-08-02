"""Database classes"""
from __future__ import annotations

import typing
from pathlib import Path

import librosa
import numpy as np
import sqlalchemy
import sqlalchemy.types as types
from praatio import textgrid
from praatio.utilities.constants import Interval
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Bundle, declarative_base, relationship


class PathType(types.TypeDecorator):
    impl = types.String

    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return Path(value)


SqlBase = declarative_base()


class File(SqlBase):
    """
    Database class for storing information about files in the corpus

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Base name of the file
    relative_path: :class:`~pathlib.Path`
        Path of the file relative to the root corpus directory
    modified: bool
        Flag for whether the file has been changed in the database for exporting
    text_file: :class:`~splaat.db.TextFile`
        TextFile object with information about the transcript of a file
    sound_file: :class:`~splaat.db.SoundFile`
        SoundFile object with information about the audio of a file
    utterances: list[:class:`~splaat.db.Utterance`]
        Utterances in the file
    """

    __tablename__ = "file"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    relative_path = Column(PathType, nullable=False)
    modified = Column(Boolean, nullable=False, default=False, index=True)
    text_file = relationship(
        "TextFile", back_populates="file", uselist=False, cascade="all, delete"
    )
    sound_file = relationship(
        "SoundFile", back_populates="file", uselist=False, cascade="all, delete"
    )
    utterances = relationship(
        "Utterance",
        back_populates="file",
        order_by="Utterance.start",
        collection_class=ordering_list("start"),
        cascade="all, delete",
        cascade_backrefs=False,
    )

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the file"""
        return len(self.utterances)

    @property
    def duration(self) -> float:
        """Duration of the associated sound file"""
        return self.sound_file.duration

    @property
    def num_channels(self) -> int:
        """Number of channels of the associated sound file"""
        return self.sound_file.num_channels

    @property
    def sample_rate(self) -> int:
        """Sample rate of the associated sound file"""
        return self.sound_file.sample_rate

    def save(self, export_root_directory=None, output_format="short_textgrid") -> None:
        """
        Output File to TextGrid.
        """

        session = sqlalchemy.orm.object_session(self)

        max_time = self.sound_file.duration
        tiers = {}

        tg = textgrid.Textgrid()
        tg.maxTimestamp = max_time
        for utterance in self.utterances:
            phone_tier_name = "phones"
            word_tier_name = "words"
            if phone_tier_name not in tiers:
                tiers[word_tier_name] = textgrid.IntervalTier(
                    word_tier_name, [], minT=0, maxT=max_time
                )
                tiers[phone_tier_name] = textgrid.IntervalTier(
                    phone_tier_name, [], minT=0, maxT=max_time
                )

            phone_intervals = (
                session.query(PhoneInterval).filter(PhoneInterval.utterance_id == utterance.id)
            ).all()
            for pi in phone_intervals:
                tiers[phone_tier_name].insertEntry(
                    Interval(
                        start=pi.start,
                        end=min(pi.end, max_time),
                        label=pi.phone,
                    )
                )
            word_intervals = (
                session.query(WordInterval).filter(WordInterval.utterance_id == utterance.id)
            ).all()
            for wi in word_intervals:
                tiers[word_tier_name].insertEntry(
                    Interval(
                        start=wi.start,
                        end=min(wi.end, max_time),
                        label=wi.word,
                    )
                )
        for t in tiers.values():
            tg.addTier(t)
        if export_root_directory is None:
            export_path = self.text_file.text_file_path
        else:
            ext = ".TextGrid"
            if output_format == "json":
                ext = ".json"
            export_path = Path(export_root_directory).joinpath(self.relative_path, self.name + ext)
            export_path.parent.mkdir(exist_ok=True, parents=True)
        tg.save(export_path, includeBlankSpaces=True, format=output_format)


class SoundFile(SqlBase):
    """

    Database class for storing information about sound files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~splaat.db.File`
    file: :class:`~splaat.db.File`
        Root file
    sound_file_path: :class:`~pathlib.Path`
        Path to the audio file
    format: str
        Format of the audio file (flac, wav, mp3, etc)
    sample_rate: int
        Sample rate of the audio file
    duration: float
        Duration of audio file
    num_channels: int
        Number of channels in the audio file
    """

    __tablename__ = "sound_file"

    file_id = Column(ForeignKey("file.id"), primary_key=True)
    file = relationship("File", back_populates="sound_file")
    sound_file_path = Column(PathType, nullable=False)
    format = Column(String, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    num_channels = Column(Integer, nullable=False)

    def normalized_waveform(
        self, start: float = 0, end: typing.Optional[float] = None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Load a normalized waveform for acoustic processing/visualization

        Parameters
        ----------
        start: float, optional
            Starting time point to return, defaults to 0
        end: float, optional
            Ending time point to return, defaults to the end of the file

        Returns
        -------
        numpy.ndarray
            Time points
        numpy.ndarray
            Sample values
        """
        if end is None or end > self.duration:
            end = self.duration

        y, _ = librosa.load(
            self.sound_file_path, sr=None, mono=False, offset=start, duration=end - start
        )
        if len(y.shape) > 1 and y.shape[0] == 2:
            y /= np.max(np.abs(y))
            num_steps = y.shape[1]
        else:
            y /= np.max(np.abs(y), axis=0)
            num_steps = y.shape[0]
        y[np.isnan(y)] = 0
        x = np.linspace(start=start, stop=end, num=num_steps)
        return x, y

    def load_audio(
        self, start: float = 0, end: typing.Optional[float] = None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Load a normalized waveform for acoustic processing/visualization

        Parameters
        ----------
        start: float, optional
            Starting time point to return, defaults to 0
        end: float, optional
            Ending time point to return, defaults to the end of the file

        Returns
        -------
        numpy.array
            Time points
        numpy.array
            Sample values
        """
        if end is None or end > self.duration:
            end = self.duration

        y, _ = librosa.load(
            self.sound_file_path, sr=16000, mono=False, offset=start, duration=end - start
        )
        return y


class TextFile(SqlBase):
    """
    Database class for storing information about transcription files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~splaat.db.File`
    file: :class:`~splaat.db.File`
        Root file
    text_file_path: :class:`~pathlib.Path`
        Path to the transcription file
    file_type: str
        Type of the transcription file (lab, TextGrid, etc)
    """

    __tablename__ = "text_file"

    file_id = Column(ForeignKey("file.id"), primary_key=True)
    file = relationship("File", back_populates="text_file")
    text_file_path = Column(PathType, nullable=False)
    file_type = Column(String, nullable=False)


class Utterance(SqlBase):
    """

    Database class for storing information about utterances

    Parameters
    ----------
    id: int
        Primary key
    start: float
        Beginning timestamp of the utterance
    end: float
        Ending timestamp of the utterance, -1 if there is no audio file
    duration: float
        Duration of the utterance
    channel: int
        Channel of the utterance in the audio file
    text: str
        Input text for the utterance
    file_id: int
        Foreign key to :class:`~splaat.db.File`
    file: :class:`~splaat.db.File`
        File object that the utterance is from
    phone_intervals: list[:class:`~splaat.db.PhoneInterval`]
        Reference phone intervals
    word_intervals: list[:class:`~splaat.db.WordInterval`]
        Aligned word intervals
    """

    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    _duration = sqlalchemy.orm.deferred(
        Column("duration", Float, sqlalchemy.Computed('"end" - "start"'), index=True)
    )
    channel = Column(Integer, nullable=False)
    text = Column(String)
    phone_text = Column(String)
    file_id = Column(Integer, ForeignKey("file.id"), index=True, nullable=False)
    file = relationship("File", back_populates="utterances", cascade_backrefs=False)
    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="utterance",
        order_by="PhoneInterval.start",
        collection_class=ordering_list("start"),
        cascade="all, delete-orphan",
    )
    word_intervals = relationship(
        "WordInterval",
        back_populates="utterance",
        order_by="WordInterval.start",
        collection_class=ordering_list("start"),
        cascade="all, delete-orphan",
    )
    other_intervals = relationship(
        "OtherInterval",
        back_populates="utterance",
        order_by="OtherInterval.start",
        collection_class=ordering_list("start"),
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        sqlalchemy.Index("utterance_position_index", "file_id", "start", "end", "channel"),
    )

    @hybrid_property
    def duration(self) -> float:
        return self.end - self.start

    @duration.expression
    def duration(cls):
        return cls._duration

    def __repr__(self) -> str:
        """String representation of the utterance object"""
        return f"<Utterance in {self.file_name} from {self.start} to {self.end}>"

    @property
    def file_name(self) -> str:
        """Name of the utterance's file"""
        return self.file.name


class PhoneInterval(SqlBase):
    """

    Database class for storing information about aligned phone intervals

    Parameters
    ----------
    id: int
        Primary key
    phone: str
        Phone label
    start: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    utterance_id: int
        Foreign key to :class:`~splaat.db.Utterance`
    utterance: :class:`~splaat.db.Utterance`
        Utterance of the interval
    word_interval_id: int
        Foreign key to :class:`~splaat.db.WordInterval`
    word_interval: :class:`~splaat.db.WordInterval`
        Word interval that is associated with the phone interval
    """

    __tablename__ = "phone_interval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone = Column(String, nullable=False, index=True)
    start = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    _duration = sqlalchemy.orm.deferred(
        Column("duration", Float, sqlalchemy.Computed('"end" - "start"'), index=True)
    )

    word_interval_id = Column(
        Integer, ForeignKey("word_interval.id", ondelete="SET NULL"), index=True, nullable=True
    )
    word_interval = relationship("WordInterval", back_populates="phone_intervals")

    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), index=True, nullable=False
    )
    utterance = relationship("Utterance", back_populates="phone_intervals")

    @hybrid_property
    def duration(self) -> float:
        return self.end - self.start

    @duration.expression
    def duration(cls):
        return cls._duration

    @property
    def label(self) -> str:
        return self.phone

    def __repr__(self):
        return f"<PhoneInterval {self.phone} from {self.start}-{self.end} for utterance {self.utterance_id}>"


class WordInterval(SqlBase):
    """

    Database class for storing information about aligned word intervals

    Parameters
    ----------
    id: int
        Primary key
    word: str
        Word label
    start: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    utterance_id: int
        Foreign key to :class:`~splaat.db.Utterance`
    utterance: :class:`~splaat.db.Utterance`
        Utterance of the interval
    phone_intervals: list[:class:`~splaat.db.PhoneInterval`]
        Phone intervals for the word interval
    """

    __tablename__ = "word_interval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String)
    start = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    _duration = sqlalchemy.orm.deferred(
        Column("duration", Float, sqlalchemy.Computed('"end" - "start"'))
    )

    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), index=True, nullable=False
    )
    utterance = relationship("Utterance", back_populates="word_intervals")

    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="word_interval",
        order_by="PhoneInterval.start",
        collection_class=ordering_list("start"),
    )

    @hybrid_property
    def duration(self) -> float:
        return self.end - self.start

    @duration.expression
    def duration(cls):
        return cls._duration

    @property
    def label(self) -> str:
        return self.word

    @property
    def confidence(self) -> typing.Optional[float]:
        return None

    def __repr__(self):
        return f"<WordInterval {self.word} from {self.start}-{self.end} for utterance {self.utterance_id}>"


class OtherInterval(SqlBase):
    """

    Database class for storing information about aligned word intervals

    Parameters
    ----------
    id: int
        Primary key
    label: str
        Interval label
    start: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    utterance_id: int
        Foreign key to :class:`~splaat.db.Utterance`
    utterance: :class:`~splaat.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "other_interval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tier_name = Column(String)
    label = Column(String)
    start = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    _duration = sqlalchemy.orm.deferred(
        Column("duration", Float, sqlalchemy.Computed('"end" - "start"'))
    )

    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), index=True, nullable=False
    )
    utterance = relationship("Utterance", back_populates="other_intervals")

    @hybrid_property
    def duration(self) -> float:
        return self.end - self.start

    @duration.expression
    def duration(cls):
        return cls._duration

    def __repr__(self):
        return f"<OtherInterval {self.label} from {self.start}-{self.end} for utterance {self.utterance_id}>"
