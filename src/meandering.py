import math
import time
import calendar

if __name__ == "__main__":

    # Constants
    n = 721

    # Generate a meandering wind, with constant standard deviations
    start_time = time.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    f = open(".\\meandering.csv", "w")
    f.write("Time.Stamp, U, V, sU, sV, cUV\n")
    for i in range(n):
        cur_time = time.gmtime(calendar.timegm(start_time) + i)
        x = 6 * math.pi * i / n
        u = math.cos(x)
        v = math.sin(x)
        f.write("%s, %f, %f, 0.3, 0.3, 0.0\n" % (cur_time, u, v))
    f.close()
