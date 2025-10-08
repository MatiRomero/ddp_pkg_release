import csv
days = [0,1,2,3,4,5,6,7]
ds   = [210, 240, 270,300]
shadows  = ["hd", "pb", "naive"]
filename = "config_rbatch_highd.csv"

with open(filename, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "d", "shadow", "jobs_csv", "save_csv"])
    for day in days:
        jobs = f"data/meituan_city_lunchtime_plat10301330_day{day}.csv"
        for D in ds:
            for sh in shadows:
                save = f"results/rbatch05_highd/meituan_day{day}_d{D}_{sh}_rbatch.csv"
                w.writerow([day, D, sh, jobs, save])

print("Wrote config.csv with", len(days)*len(ds)*len(shadows), "rows.")