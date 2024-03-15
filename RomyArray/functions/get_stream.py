def __get_stream(seed, tbeg, tend, repository="online"):

    st = obs.Stream()

    invs = []

    print(f" -> loading {seed}...")
    try:
        st0, inv0 = __querrySeismoData(
                                    seed_id=seed,
                                    starttime=tbeg-10,
                                    endtime=tend+10,
                                    repository=repository,
                                    path=None,
                                    restitute=False,
                                    detail=None,
                                    fill_value=None,
                                    )

        st0 = st0.remove_response(inv0, output="VEL", water_level=60)

        st0 = st0.rotate('->ZNE', inventory=inv0)

        st0 = st0.trim(tbeg, tend)

        if len(st0) != 0:
            st += st0

    except Exception as e:
        print(e)
        print(f" -> failed to load data: {seed}")

    return st, invsdef __get_stream(seed, tbeg, tend, repository="online"):

    st = obs.Stream()

    invs = []

    print(f" -> loading {seed}...")
    try:
        st0, inv0 = __querrySeismoData(
                                    seed_id=seed,
                                    starttime=tbeg-10,
                                    endtime=tend+10,
                                    repository=repository,
                                    path=None,
                                    restitute=False,
                                    detail=None,
                                    fill_value=None,
                                    )

        st0 = st0.remove_response(inv0, output="VEL", water_level=60)

        st0 = st0.rotate('->ZNE', inventory=inv0)

        st0 = st0.trim(tbeg, tend)

        if len(st0) != 0:
            st += st0

    except Exception as e:
        print(e)
        print(f" -> failed to load data: {seed}")

    return st, invs