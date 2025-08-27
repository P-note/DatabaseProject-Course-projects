def baseline(input_file):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(',') for line in lines]
    
    skyline = []
    not_skyline = []
        
    for id_a, city_a, quality_a, _, price_a in lines:
        dominated = False
        for id_b, city_b, quality_b, _, price_b in lines:
            if id_a == id_b:
                continue
            # Checks whether a is dominated by b
            # quality feature should be maximized and price feature should be minimized
            if (int(quality_a) <= int(quality_b) and int(price_a) >= int(price_b)) and (int(quality_a) < int(quality_b) or int(price_a) > int(price_b)):
                dominated = True
                break
        if not dominated:
            skyline.append((id_a, city_a, quality_a, price_a))
        else:
            not_skyline.append((id_a, city_a, quality_a, price_a))
    
    return skyline, not_skyline


def pareto_optimal(input_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(',') for line in lines]

    skyline = []
    not_skyline = []
    
    ############# EDIT HERE #############

    for id_a, city_a, quality_a, service_a, price_a in lines:
        dominated = False
        for id_b, city_b, quality_b, service_b, price_b in lines:
            if id_a == id_b:
                continue
            # Checks whether a is dominated by b
            # quality feature should be maximized and price feature should be minimized
            if (int(quality_a) <= int(quality_b) and int(price_a) >= int(price_b) and int(service_a) <= int(service_b)) and (int(quality_a) < int(quality_b) or (int(price_a) > int(price_b)) or (int(service_a) < int(service_b))) or city_a == 'Seoul':
                dominated = True
                break

    ############# EDIT HERE #############

        if not dominated:
            skyline.append((id_a, city_a, quality_a, service_a, price_a))
        else:
            not_skyline.append((id_a, city_a, quality_a, service_a, price_a))
    
    return skyline, not_skyline