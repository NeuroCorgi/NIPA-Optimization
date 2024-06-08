import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

def show_covid_cases(data, regions):
    fig, ax = plt.subplots(figsize=(17.5, 7.5))
    # dates = [dt.datetime.strptime(date, "%d-%m-%Y").date() for date in data["date"]]

    for region in regions:
        ax.plot(data['k'], data[region], label=region.title())

    # ax.plot(data['k'], data['AGUASCALIENTES'], label=regions[0].title())

    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    plt.legend()
    plt.show()

def show_covid_cases_of_region(data, region):
    fig, ax = plt.subplots(figsize=(17.5, 7.5))
    # dates = [dt.datetime.strptime(date, "%d-%m-%Y").date() for date in data["date"]]

    ax.plot(data["k"], data[str(region)], label=region)

    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    plt.legend()
    plt.show()