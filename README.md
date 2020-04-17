# Odd-tomatoes-vs-Rotten-tomatoes-web-application-using-Flask

# FLASH APPLICATION #

### Aim ###

* Making a Flask application so user can upload their photos and receive the prediction.

### Description ###

* This is a simple Flask App that can classify an image as Odd or Rotten tomatoes using a convolutional neural network model. This application has Front end made with HTML and back end made with FLASK

### How to use ###

* In order to run our model on a Flask application locally, you need to clone this repository and then set up the environment by these    following commands:

* On the Terminal, use these commands:

```
    export FLASK_APP=app.py
    flask run --host=0.0.0.0
```

* On the Anaconda Prompt, use these commands :

```
    python app.py
    
```

* Then browse to this URL http://localhost:5000/


### Home Page ###

<p align="center">

<img src='static/images/1.png'>

</p>

### Example of results ###

<p align="center">

<img src='static/images/2.png'>
<img src='static/images/3.png'>

</p>

### Conclusion ###

The Architecture and parameter used in this network are capable of producing accuracy of 94.98% on Validation Data which is pretty good. You can download this trained model from resource directory and Play with it.

