# VFL playground

This repo contains a trial of vertical federated learning (VFL)
where the data holders do not train neural networks
on their own devices.

## Why?

The purpose of this is to demonstrate that VFL
is a useful paradigm even for "simple" problems
for which neural networks are not required.
In this demo,
each data holder owns some part of the "Titanic" dataset,
which is simple enough to achieve high accuracies
even with `O(100)` datapoints.
Each data holder trains a logistic regression model
on their part of the dataset.
They send their predictions to a centralised
computational server,
which trains a neural network on the concatenation
of the outputs from each data holder
in order to better predict labels for the datapoints.
The idea behind this process is that data holders
will perform differently relative to one another
based on the specific characteristics of their own data.
Mapping these outputs to the more correct function of the data
is a non-linear process (hence why we need the neural networks!)

## Get started

### Python

This demo has been coded using python `3.8`,
but similar minor versions will work.

### Environment

Very simple - only a few packages required (and no GPUs!).
Run `pip install -r requirements.txt` to install necessary packages.

## How to run

Run `main.sh`.
This trains a model in a centralised setting
and then a model in the VFL setting.

Alternatively,
execute `python scripts/run_(de)centralised.py`,
where `(de)` is optional,
to run one of the two scripts on its own.

## Security implications

Incoming

## License

Apache 2.0.
See the [license](LICENSE) for more information.
