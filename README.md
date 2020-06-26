<small>created by [wesley beckner](http://wesleybeckner.github.io)</small>

# Product Margin Dashboard

This is an [product characterization dashboard](https://gendorf.herokuapp.com) demo using [Dash](https://plot.ly/products/dash/) 

## Getting Started

### Running the app locally

First create a virtual environment with conda (or venv) and activate it.

```

conda create -n <your_env_name> python==3.7
source activate <your_env_name>

```

Clone the git repo, then install the requirements with pip

```

git clone https://github.com/wesleybeckner/gendorf.git
cd gendorf
pip install -r requirements.txt

```

Run the app

```

python app.py

```

## About the app

This is an interactive app to assess product margins and subsequent impact on annualized EBIT for manufacturing


## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
