import setuptools
with open("README.md", "r",encoding='utf-8') as f:
    long_description = f.read()
    
    
__version__ = "0.0.1"


REPO_NAME = "UAE-USED-CAR-PRICE-PREDICTION"
Author_USER_NAME = "VANSHKAUSHIK"
SRC_REPO = "used_car_price_prediction"
Author_EMAIL = "Vansh.k0907@gmail.com"

setuptools.setup(
    Name = SRC_REPO,
    version = __version__,
    author = Author_USER_NAME,
    author_email = Author_EMAIL,
    Description = "A package for predicting used car prices in UAE",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{Author_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug_Tracker" : f"http://github.com/{Author_USER_NAME}/{REPO_NAME}/issues",},
    Package_dir = {"":"src"},
    Packages = setuptools.find_packages(where="src"),   
)