"""Names used by the benchmark, with known mishearings on phone audio.

Variants are how recognizers plausibly mishear the name. They drive the mock
backends and serve as hard distractors in rosters. Respellings help TTS
pronounce the name when generating caller audio.
"""
from __future__ import annotations

# name: (sound-alike variants, TTS respelling or None)
NAMES: dict[str, tuple[list[str], str | None]] = {
    "Siobhan Murphy": (["Sharon Murphy", "Shivon Murphy"], "Shivawn Murphy"),
    "Aoife Byrne": (["Eva Byrne", "Ifa Burn"], "Eefa Byrne"),
    "Saoirse Kelly": (["Sersha Kelly", "Sara Kelly"], "Seersha Kelly"),
    "Nguyen Tran": (["Win Tran", "Newin Tran"], "Win Tran"),
    "Subramaniam Iyer": (["Subramanian Iyer", "Subramaniam Ayer"], None),
    "Krzysztof Nowak": (["Christoph Nowak", "Kristof Novak"], "Kshishtof Novak"),
    "Oluwaseun Adeyemi": (["Oluwasean Adeyemi", "Olu Adeyemi"], "Oloowa-shay-oon Adeyemi"),
    "Fahad Rahman": (["Fathom Rahman", "Fahd Rahman"], None),
    "Joaquin Ortiz": (["Walking Ortiz", "Joachim Ortiz"], "Wah-keen Ortiz"),
    "Priyanka Reddy": (["Priyanca Reddy", "Bianca Reddy"], None),
    "Xiaoming Chen": (["Shaoming Chen", "Showming Chen"], "Shyow-ming Chen"),
    "Mohammed Al-Sayed": (["Mohamed Al Said", "Muhammad Al-Sayed"], None),
    "Siddharth Rao": (["Sid Hart Rao", "Siddhart Rao"], None),
    "Thandiwe Ndlovu": (["Tandy Ndlovu", "Thandiwe Lovu"], "Tan-dee-way Ndlovu"),
    "Ewa Kowalczyk": (["Eva Kowalczyk", "Ewa Kovalchik"], "Eva Kovalchik"),
    "Niamh Walsh": (["Neve Walsh", "Nia Walsh"], "Neev Walsh"),
    "Sean O'Brien": (["Shawn O'Brien", "Sean Brian"], None),
    "Catherine Lee": (["Kathryn Lee", "Katherine Lee"], None),
    "Jon Smith": (["John Smith", "Jonathan Smith"], None),
    "Sri Harsha Mudumba": (["Sree Harsha Mudumba", "Sri Harsha Madumba"], None),
    "Rhys Evans": (["Reese Evans", "Rice Evans"], "Reess Evans"),
    "Ngozi Okafor": (["Gozi Okafor", "Nogozi Okafor"], "N-go-zee Okafor"),
    "Yusuf Demir": (["Joseph Demir", "Yousef Demir"], None),
    "Ines Duarte": (["Agnes Duarte", "Inez Duarte"], "Ee-nesh Duarte"),
}

FIRST_TIME_NAMES: dict[str, tuple[list[str], str | None]] = {
    "Tadhg Brennan": (["Tiger Brennan", "Tie Brennan"], "Tyg Brennan"),
    "Grainne Doyle": (["Grania Doyle", "Granny Doyle"], "Grawn-ya Doyle"),
    "Chiamaka Eze": (["Chimaka Eze", "Kiamaka Eze"], None),
    "Wojciech Zielinski": (["Voytek Zielinski", "Wojtek Zelinski"], "Voy-chek Zhelinski"),
    "Anh Pham": (["Ann Pham", "On Fam"], None),
    "Bartholomew Reyes": (["Bartholomew Rays", "Bart Reyes"], None),
}
