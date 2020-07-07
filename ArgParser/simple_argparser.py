import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-n", "--name", required=True, help="name of the author")
args = vars(ap.parse_args())

# display the arguments
print("Hi there {}, it's nice to meet you!".format(args["name"]))