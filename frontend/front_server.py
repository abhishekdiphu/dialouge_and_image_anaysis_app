
from flask import Flask, render_template, request, jsonify
from itertools import zip_longest

from flask_cors import CORS


print("Model successfully loaded!")

data_for_analysis = {"human" : [], "ai" : []}




app = Flask("ML frontend server")
CORS(app)


@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application
        page over the Flask channel
    '''
    return render_template('index.html')




if __name__ == "__main__":

    
    app.run(host="0.0.0.0", port=9000)
