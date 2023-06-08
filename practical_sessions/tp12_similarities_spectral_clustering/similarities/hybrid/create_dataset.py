"""
    Create csv with hybrid data
    (both numerical and categorical attributes)
"""

import csv

with open("hybrid_data.csv", "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(["Name", "Fifa note", "speed", "favorite meal", "nationality", "world cups"])
    filewriter.writerow(["Toni Kroos", "90", "70", "pasta", "German", "1"])
    filewriter.writerow(["David De Gea", "91", "70", "fries", "Spanish", "0"])
    filewriter.writerow(["Sergio Ramos", "91", "80", "pasta", "Spanish", "1"])
    filewriter.writerow(["Kilyan Mbappé", "100", "100", "fries", "French", "1"])
    filewriter.writerow(["Mohammed Salah", "88", "95", "pasta", "Egyptian", "0"])
    filewriter.writerow(["Mats Hummels", "80", "73", "vegetables", "German", "1"])
