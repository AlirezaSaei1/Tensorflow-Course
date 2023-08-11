import csv

time_step = []
sunspots = []

with open('TimeSeries/Sunspots.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
