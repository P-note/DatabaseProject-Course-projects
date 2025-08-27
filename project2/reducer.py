#!/usr/bin/env python3

import sys

lines = []
for line in sys.stdin:
    line = line.strip().split()
    lines.append(line)

########## EDIT HERE ##########

skyline = []
not_skyline = []

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

    if not dominated:
        skyline.append((id_a, city_a, quality_a, service_a, price_a))
    else:
        not_skyline.append((id_a, city_a, quality_a, service_a, price_a))


########## EDIT HERE ##########

for point in skyline:
    print("{0} {1} {2} {3} {4}".format(point[0], point[1], point[2], point[3], point[4]))