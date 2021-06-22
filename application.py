from model import predict_toxic
from sys import argv
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import redirect




app = Flask(__name__) #This is where everything starts. Look from this point
#template - html /flamework
#statis - css - thing to make it beautiful
# here we construct a Flask object and __name__ sets the script to the root


#first landing place
@app.route('/')
def index():  
    return render_template('index.html')
    

@app.route('/toxic')
def classification():
    print(request.args)
    user_input = request.args['user_input']
    print(user_input)
    pred = predict_toxic(user_input)
    return render_template('toxic_class.html', pred=pred)


@app.route('/algorithm')
def algorithm():
    return render_template('algorithm.html')



if __name__ == '__main__':
    app.run(debug=True, port=5000)   

