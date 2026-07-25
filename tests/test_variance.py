from flight_routes.variance import haversine_nm


def test_haversine_nm_egll_kjfk_great_circle():
    # London Heathrow to JFK: known great-circle distance is ~2991 nm
    dist = haversine_nm(51.4700, -0.4543, 40.6413, -73.7781)
    assert 2985 <= dist <= 2997


def test_haversine_nm_zero_distance():
    assert haversine_nm(51.47, -0.4543, 51.47, -0.4543) == 0
