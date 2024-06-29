import polars as pl
import datetime as dt

from utils.Country import Country
from utils.io import save_train_data

def compensate_fluctuations(data, dates, time):
    """
    Compensate for fluctuations in the data using a rolling mean.

    Parameters
    ----------
    data : DataFrame
        The input data containing infection numbers.
    dates : dict
        A dictionary mapping indices to dates.
    time : int
        The window size in days for the rolling mean.

    Returns
    -------
    DataFrame
        A DataFrame with compensated fluctuations.
    """
    # Transpose data to work with dates and regions
    I_transposed = data.transpose(include_header=True, header_name="k", column_names="regions")
    start_date = dates[1]

    col = []
    for row in I_transposed[:, 0].to_list():
        date = start_date + dt.timedelta(days=int(row)-1)
        col.append(date)

    I_transposed = I_transposed.with_columns(pl.Series("k", col))

    # Apply rolling mean to compensate for fluctuations
    I_compensated = I_transposed.select([pl.col(col.name).rolling_mean_by('k', window_size=f'{time}d', closed='right') for col in I_transposed[:, 1:].get_columns()])
    I_compensated = I_compensated.insert_column(0, I_transposed.get_column('k'))

    col = []
    for row in I_compensated[:, 0].to_list():
        k = int((row - start_date).days) + 1
        col.append(str(k))

    I_compensated = I_compensated.with_columns(pl.Series("k", col))
    I_compensated = I_compensated.transpose(include_header=True, header_name="regions", column_names="k")
    I = I_compensated

    return I

def get_population(regions, population_data):
    """
    Get the population of each region.

    Parameters
    ----------
    regions : list of str
        The list of region names.
    population_data : DataFrame
        The DataFrame containing the population data for the regions.

    Returns
    -------
    DataFrame
        A DataFrame with the population of each region.
    """
    population_per_region = {region: population_data.to_list()[regions.index(region)] for region in regions}

    population = pl.DataFrame({
        "regions": regions,
        "population": population_per_region.values()
    })
    save_train_data(population, Country.Mexico, "population")

    return population

def get_I(data, regions, population, final_date):
    """
    Get the fraction of COVID-19 cases of each region.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing COVID-19 infection data.
    regions : list of str
        The list of region names.
    population : DataFrame
        The DataFrame containing population data for the regions.
    final_date : datetime.date
        The final date up to which to retrieve data.

    Returns
    -------
    tuple
        A tuple containing the DataFrame with infection fractions and a dictionary with dates.
    """
    I = pl.DataFrame({
        "regions": regions
    })

    population_per_region = {region: population for region, population in zip(regions, population.get_column('population').to_list())}

    k = 1
    I_dates = {}

    # for each region in the data
    for col in data.get_columns():
        header = col.name

        try:
            date = dt.datetime.strptime(header, "%d-%m-%Y").date()

            if date <= final_date:
                I_dates[k] = date
                # If the date is before the end date, insert the fraction of COVID-19 cases of each region
                I.insert_column(k, pl.Series(str(k),
                                       [(infections / population_per_region[regions[i]]) for i, infections in enumerate(col.to_list())]))
                k += 1
            else:
                break
        # If the header is not a date, continue
        except ValueError:
            continue
    
    k_list = list(I_dates.keys())
    print(f"First date: {I_dates[k_list[0]]} Last date: {I_dates[k_list[-1]]} Difference in days: {(I_dates[k_list[-1]] - I_dates[k_list[0]])}")

    return I, I_dates

# Get the data for the specified country
def get_data(country, final_date="all", compensate_fluctuations=True):
    """
    Get COVID-19 data for the specified country.

    Parameters
    ----------
    country : Country
        The country for which to retrieve data.
    final_date : str or datetime.date, optional
        The final date up to which to retrieve data. Default is "all".
    compensate_fluctuations : bool, optional
        Whether to compensate for fluctuations in the data. Default is True.

    Returns
    -------
    tuple
        A tuple containing the data, dates, and regions.
    """
    match country:
        case Country.Mexico:
            return get_mexican_states(final_date, compensate_fluctuations)
        case Country.Netherlands:
            return get_netherlands_provinces(final_date, compensate_fluctuations)
        case Country.Hubei:
            return get_hubei_province(final_date, compensate_fluctuations)

def get_hubei_province(final_date="all", do_compensate_fluctuations=True):
    """
    Get COVID-19 data for Hubei province.

    Parameters
    ----------
    final_date : str or datetime.date, optional
        The final date up to which to retrieve data. Default is "all".
    do_compensate_fluctuations : bool, optional
        Whether to compensate for fluctuations in the data. Default is True.

    Returns
    -------
    tuple
        A tuple containing the DataFrame with infection fractions, a dictionary with dates, 
        and a list of region names.
    """
    if final_date == "all":
        final_date = dt.datetime(year=2020, month=2, day=14).date()

    start_date = dt.datetime(year=2020, month=1, day=21).date()

    DATA_FILENAME = "data/hubei_data.csv"
    full_data = pl.read_csv(DATA_FILENAME)
    full_data = full_data.filter(pl.col("City") != "Shennongjia")

    full_data = full_data.rename({"Population": "population",
                                  "City": "regions"})
    
    for col in full_data[:, 2:].get_columns():
        old_name = col.name
        new_name = dt.datetime.strptime(old_name + " 2020", "%d-%b %Y").strftime("%d-%m-%Y")
        full_data = full_data.rename({old_name: new_name})

    regions = full_data.get_column('regions').to_list()
    population = full_data[:, :2]

    # Average out the 8th step of the data
    eight_row = []
    for row in range(len(full_data.to_series(9).to_list())):
        value_seventh = full_data.to_series(8).to_list()[row]
        value_nineth = full_data.to_series(10).to_list()[row]

        eight_row.append(int((value_seventh + value_nineth) / 2))

    full_data = full_data.with_columns(pl.Series("28-01-2020", eight_row))

    I, I_dates = get_I(full_data, regions, population, final_date)

    if do_compensate_fluctuations:
        I = compensate_fluctuations(I, I_dates, 5)

    save_train_data(I, Country.Hubei, "I")

    return I, I_dates, regions

def get_mexican_states(final_date="all", do_compensate_fluctuations=True):
    """
    Receive the COVID-19 cases from the Mexican states.

    Parameters
    ----------
    final_date : str or datetime.date, optional
        The final date up to which to retrieve data. Default is "all".
    do_compensate_fluctuations : bool, optional
        Whether to compensate for fluctuations in the data. Default is True.

    Returns
    -------
    tuple
        A tuple containing the DataFrame with infection fractions, a dictionary with dates, 
        and a list of region names.
    """
    if final_date == "all":
        final_date = dt.datetime(year=2023, month=6, day=24).date()

    DATA_FILENAME = "data/mexican_state_data.csv"
    full_data = pl.read_csv(DATA_FILENAME)

    # Remove the total national cases
    full_data = full_data.filter(pl.col("nombre") != "Nacional")

    # Capitalize the region names
    capitalize_regions = lambda regions: [region.title() for region in regions]
    full_data = full_data.with_columns(pl.Series("nombre", 
                                        capitalize_regions(full_data.get_column('nombre').to_list())))
    
    full_data = full_data.sort("cve_ent")
    regions = full_data.get_columns()[2].to_list()

    # Get the population of each region and save it to a csv file
    population = get_population(regions, full_data.get_column('poblacion'))

    # Get the fraction of COVID-19 cases of each region
    I, I_dates = get_I(full_data, regions, population, final_date)

    if do_compensate_fluctuations:
        I = compensate_fluctuations(I, I_dates, 7)

    save_train_data(I, Country.Mexico, "I")

    return I, I_dates, regions

def get_netherlands_provinces(final_date="all"):
    """
    Get COVID-19 data for the Netherlands provinces.

    Parameters
    ----------
    final_date : str or datetime.date, optional
        The final date up to which to retrieve data. Default is "all".
    do_compensate_fluctuations : bool, optional
        Whether to compensate for fluctuations in the data. Default is True.

    Returns
    -------
    tuple
        A tuple containing the DataFrame with infection fractions, a dictionary with dates, 
        and a list of region names.
    """
    # Placeholder for future implementation
    pass

def get_NIPA_data(data, dates, n=875, pred_days=3, over_days=-1):
    """
    Get data for Original NIPA (Network-inference-based prediction of the COVID-19 epidemic outbreak).

    Parameters
    ----------
    data : DataFrame
        The input data containing infection numbers.
    dates : dict
        A dictionary mapping indices to dates.
    n : int, optional
        The number of days to use for training. Default is 875.
    pred_days : int, optional
        The number of days to predict. Default is 3.
    over_days : int, optional
        The number of days to overlap for training. Default is -1.

    Returns
    -------
    tuple
        A tuple containing the training data, prediction data, and training dates.
    """
    if (n+pred_days) > len(data.get_columns()):
        raise ValueError("Amount of dates is greater than the data available! Please choose a smaller n or smaller prediction days.")
    
    date_values = [date for date in dates.values()]
    X = {}
    Y = {}
    train_dates = {}

    print(f"Amount of dates: {len(date_values)} N: {n} Prediction days: {pred_days} Over days: {over_days}")

    if over_days == -1:
        # Prepare data for all days except the last pred_days
        X[dates[n]] = data[:, :(n+1)]           # Train data
        Y[dates[n]] = data[:, n:n+pred_days+1]  # Prediction data

        curr_dates = date_values[:n + pred_days]
        train_dates[1] = curr_dates
    else:
        for i in range(over_days, len(date_values)-pred_days+1):
            X[dates[i]] = data[:, :i]             # Train data
            Y[dates[i]] = data[:, i:i+pred_days]  # Prediction data

            curr_dates = date_values[:i + pred_days]  
            train_dates[i+1-over_days] = curr_dates

    return X, Y, train_dates

def get_static_NIPA_data(data, dates, n=875, train_days=20, pred_days=3):
    """
    Get data for Static NIPA (Comparing the accuracy of several network-based COVID-19 prediction algorithms).

    Parameters
    ----------
    data : DataFrame
        The input data containing infection numbers.
    dates : dict
        A dictionary mapping indices to dates.
    n : int, optional
        The number of days to use for training. Default is 875.
    train_days : int, optional
        The number of days to train the model. Default is 20.
    pred_days : int, optional
        The number of days to predict. Default is 3.

    Returns
    -------
    tuple
        A tuple containing the training data, prediction data, and training dates.
    """
    # Placeholder for future implementation
    pass

def get_dynamic_NIPA_data(data, dates, n=875, train_days=20, pred_days=3):  
    """
    Get data for Dynamic NIPA (Comparing the accuracy of several network-based COVID-19 prediction algorithms).

    Parameters
    ----------
    data : DataFrame
        The input data containing infection numbers.
    dates : dict
        A dictionary mapping indices to dates.
    n : int, optional
        The number of days to use for training. Default is 875.
    train_days : int, optional
        The number of days to train the model. Default is 20.
    pred_days : int, optional
        The number of days to predict. Default is 3.

    Returns
    -------
    tuple
        A tuple containing the training data, prediction data, and training dates.
    """
    # Placeholder for future implementation
    pass
