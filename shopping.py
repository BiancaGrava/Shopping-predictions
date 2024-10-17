import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence_list = []
    label_list = []

    with open(filename, 'r') as file:
        # Create a CSV reader object using space as a delimiter
        csv_reader = csv.reader(file, delimiter=',')

        # Skip the header row
        next(csv_reader)

        # Iterate over the remaining rows
        for row in csv_reader:
            Administrative = int(row[0])
            Administrative_Duration = float(row[1])
            Informational = int(row[2])
            Informational_Duration = float(row[3])
            ProductRelated = int(row[4])
            ProductRelated_Duration = float(row[5])
            BounceRates = float(row[6])
            ExitRates = float(row[7])
            PageValues = float(row[8])
            SpecialDay = float(row[9])
            Month = None
            if row[10] == "Jan":
                Month = 0
            elif row[10] == "Feb":
                Month = 1
            elif row[10] == "Mar":
                Month = 2
            elif row[10] == "Apr":
                Month = 3
            elif row[10] == "May":
                Month = 4
            elif row[10] == "June":
                Month = 5
            elif row[10] == "Jul":
                Month = 6
            elif row[10] == "Aug":
                Month = 7
            elif row[10] == "Sep":
                Month = 8
            elif row[10] == "Oct":
                Month = 9
            elif row[10] == "Nov":
                Month = 10
            elif row[10] == "Dec":
                Month = 11
            OperatingSystems = int(row[11])
            Browser = int(row[12])
            Region = int(row[13])
            TrafficType = int(row[14])
            VisitorType1 = row[15]
            if VisitorType1 == "Returning_Visitor":
                VisitorType = int(1)
            else:
                VisitorType = int(0)
            Weekend1 = row[16]
            if Weekend1 == "TRUE":
                Weekend = int(1)
            else:
                Weekend = 0
            Label1 = row[17]
            if Label1 == "TRUE":
                Label = int(1)
            else:
                Label = int(0)

            #make evidence list:
            current_evidence=[Administrative,Administrative_Duration,Informational,Informational_Duration,ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,VisitorType,Weekend]
            evidence_list.append(current_evidence)
            label_list.append(Label)

        resulting_tuple = (evidence_list,label_list)
        return resulting_tuple


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    true_positive = 0
    total_positive = 0
    true_negative = 0
    total_negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            if predicted == 1:
                true_positive += 1
            total_positive += 1
        else:
            if predicted == 0:
                true_negative += 1
            total_negative += 1

    sensitivity = true_positive/total_positive
    specifity = true_negative/total_negative

    return (sensitivity, specifity)


if __name__ == "__main__":
    main()
