"""
Create csv with complex data
"""

import csv

file_name = 'complex_data.csv'

with open('csv_files/' + file_name, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['Name', 'Fifa note', 'speed', 'favorite meal'])
    filewriter.writerow(['Toni Kroos', '90', '80', 'pasta'])
    filewriter.writerow(['David De Gea', '91', '70', 'fries'])
    filewriter.writerow(['Sergio Ramos', '91', '75', 'pasta'])
    filewriter.writerow(['Kikyan Mbappé', '100', '100', 'fries'])
    filewriter.writerow(['Mohammed Salah', '88', '95', 'pasta'])
    filewriter.writerow(['Mats Hummels', '89', '73', 'vegetables'])
