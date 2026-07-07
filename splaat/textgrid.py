from __future__ import annotations

import pathlib

import soundfile as sf
import sqlalchemy
from praatio import textgrid as tgio

from splaat.db import (
    File,
    OtherInterval,
    PhoneInterval,
    SoundFile,
    TextFile,
    Utterance,
    WordInterval,
)


def parse_textgrid(path: pathlib.Path):
    tg = tgio.openTextgrid(path, includeEmptyIntervals=False)
    for tier_name in tg.tierNames:
        if "word" in tier_name:
            word_tier_name = tier_name
        elif "phone" in tier_name:
            phone_tier_name = tier_name

    word_intervals = []
    for wi in tg._tierDict[word_tier_name].entries:
        phone_intervals = []
        for pi in tg._tierDict[phone_tier_name].entries:
            if pi.end <= wi.start:
                continue
            if pi.start >= wi.end:
                break
            phone_intervals.append(pi)
        word_intervals.append((wi, phone_intervals))
    other_intervals = {}
    for tier_name in tg.tierNames:
        if tier_name in [word_tier_name, phone_tier_name]:
            continue
        other_intervals[tier_name] = []
        for oi in tg._tierDict[tier_name].entries:
            other_intervals[tier_name].append(oi)
    return word_intervals, other_intervals


def parse_file_to_db(
    session: sqlalchemy.orm.Session, path: pathlib.Path, root_directory: pathlib.Path = None
):
    try:
        info = sf.info(path)
    except Exception:
        return
    text_path = path.with_suffix(".TextGrid")
    text_type = "TextGrid"
    if not text_path.exists():
        return
    word_intervals, other_intervals = parse_textgrid(text_path)
    relative_path = ""
    if root_directory is not None:
        relative_path = path.relative_to(root_directory).parent
    file = File(name=path.stem, relative_path=relative_path)
    text_file = TextFile(file=file, text_file_path=text_path, file_type=text_type)
    sound_file = SoundFile(
        file=file,
        sound_file_path=path,
        format=info.format,
        sample_rate=info.samplerate,
        duration=info.duration,
        num_channels=info.channels,
    )
    session.add(file)
    session.add(text_file)
    session.add(sound_file)
    text = []
    phones = []
    utterance = Utterance(start=0, end=info.duration, channel=0, file=file, text="", phone_text="")
    for wi, phone_intervals in word_intervals:
        wi_obj = WordInterval(word=wi.label, start=wi.start, end=wi.end)
        text.append(wi.label)
        utterance.word_intervals.append(wi_obj)
        for pi in phone_intervals:
            pi_obj = PhoneInterval(phone=pi.label, start=pi.start, end=pi.end)
            wi_obj.phone_intervals.append(pi_obj)
            phones.append(pi.label)
            utterance.phone_intervals.append(pi_obj)
    utterance.text = " ".join(text)
    utterance.phone_text = " ".join(phones)
    for tier_name, intervals in other_intervals.items():
        for oi in intervals:
            oi_obj = OtherInterval(tier_name=tier_name, label=oi.label, start=oi.start, end=oi.end)
            utterance.other_intervals.append(oi_obj)
    session.add(utterance)
