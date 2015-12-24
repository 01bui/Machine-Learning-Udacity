#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    from sklearn.metrics import mean_squared_error
    cleaned_data = []

    ### your code goes here
    error = abs(predictions - net_worths)
    cleaned_data = zip(ages, net_worths, error)
    cleaned_data.sort(key=lambda tup: tup[2])
    cleaned_data = cleaned_data[0:80]
    
    return cleaned_data

