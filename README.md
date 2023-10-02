<h1 align='center'>
The UCL models for lung cancer risk prediction
</h1>

The UCL models are designed to be used to predict the risk of lung cancer (occurrence or death) in an individual who has smoked at least 100 cigarettes in their lifetime and are aged at least 40 years old.

For more details, please see the following paper: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1004287

Before using the models, please read the disclaimer. 
These models are freely available for non-profit use. Please see the Licence for more details.

## Using the model code

### :rocket: Installation

Clone this Github repository. In your terminal, navigate to the folder and subsequently run:

```bash
$ conda env create --file environment.yml
```

This will create a new conda environment called ucl_lung_cancer_models. Activate this environment as follows:

```bash
$ conda activate ucl_lung_cancer_models
```

### Get predictions

```python
import pandas as pd
import cloudpickle

def load_model_from_file(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)
    
def load_model(model='ucld'):
    return load_model_from_file(f"models/{model}/fittedmodel.p")
    
X = pd.DataFrame({
	age = [50,65,70],
	smoking_duration = [30,38,50],
	pack_years=[15,38,75]  
	})
 
 ucld_model = load_model()
 predictions = ucld_model.predict_proba(X)
 ```
 

