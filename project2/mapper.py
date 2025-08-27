#!/usr/bin/env python3

import sys

lines = []
for line in sys.stdin:
    ########## EDIT HERE ##########
    line = line.strip()
    tuple_list = line.split(",")
    id = tuple_list[0]
    city = tuple_list[1]
    quality = tuple_list[2]
    service = tuple_list[3]
    price = tuple_list[4]
    
    print('{0} {1} {2} {3} {4}'.format(id, city, quality, service, price))

    ########## EDIT HERE ##########