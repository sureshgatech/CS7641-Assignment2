
def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 0.5 else 1))
            instances.append(instance)

    return instances


def main():
    """Run this experiment"""
    training_ints = initialize_instances('E:\CS7641\Assignment2\SureshCode\data\wine_trg.csv')
    testing_ints = initialize_instances('E:\CS7641\Assignment2\SureshCode\data\wine_trg.csv')
    validation_ints = initialize_instances('E:\CS7641\Assignment2\SureshCode\data\wine_trg.csv')


if __name__ == "__main__":
    main()