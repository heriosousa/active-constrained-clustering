def get_constraints_from_neighborhoods(neighborhoods):
    ml = []

    for neighborhood in neighborhoods:
        for i in neighborhood:
            for j in neighborhood:
                if i != j and (j, i) not in ml:
                    ml.append((i, j))

    cl = []
    for neighborhood in neighborhoods:
        for other_neighborhood in neighborhoods:
            if neighborhood != other_neighborhood:
                for i in neighborhood:
                    for j in other_neighborhood:
                        if (j, i) not in cl:
                            cl.append((i, j))

    return ml, cl