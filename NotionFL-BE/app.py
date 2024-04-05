import base64
import threading
import os
import yaml
import json
import uuid
from io import BytesIO
from flask import Flask, jsonify, request, send_file, url_for
from flask_cors import CORS
from subprocess import Popen, PIPE
from datetime import datetime
from Database.auth import register_user, check_user_credentials
from flask_jwt_extended import JWTManager, create_access_token
from routes.auth_routes import auth_bp
from dotenv import load_dotenv
from mongoengine import connect
import logging

# load env varibales
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:5173"}})

# Setup database connection
db_uri = os.environ.get("MONGODB_URI")
db_name = os.environ.get("DB_NAME")
connect(db_name, host=db_uri)

# Setup Flask-JWT-Extended
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')
jwt = JWTManager(app)

# API endpoints
app.register_blueprint(auth_bp, url_prefix='/auth')


if __name__ == '__main__':
    app.run(debug=True)
