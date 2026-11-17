import pandas as pd

from buaro import future


def panel_with_partial_summer():
    rows = []
    for geo in ("EL301", "EL302"):
        for year in (2019, 2022, 2023):
            for week in range(1, 53):
                pandemic = year == 2022 and week <= 26
                rows.append({"geo": geo, "year": year, "isoweek": week, "deaths": 10.0, "pandemic": pandemic})
    return pd.DataFrame(rows)


def test_partial_pandemic_summer_is_left_out_everywhere():
    panel = panel_with_partial_summer()
    assert future.full_summers(panel) == [2019, 2023]
    assert future.full_summers(panel, (26, 36)) == [2019, 2023]
    assert future.full_summers(panel, (27, 36)) == [2019, 2022, 2023]


def test_year_ranges():
    assert future.year_ranges([2015, 2016, 2017, 2018, 2019, 2023, 2024, 2025]) == "2015 to 2019 and 2023 to 2025"
    assert future.year_ranges([2015, 2017]) == "2015 and 2017"
    assert future.year_ranges([2015, 2016, 2018, 2020, 2021]) == "2015 to 2016, 2018 and 2020 to 2021"
    assert future.year_ranges([]) == ""
