from flask import Blueprint, request, jsonify, make_response
from Database.auth import register_user, check_user_credentials
from flask_jwt_extended import create_access_token

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if register_user(data['username'], data['email'], data['password'], data['role'], data.get('organization', ''), data.get('contact', '')):
        return jsonify({'message': 'User registered successfully'}), 201
    else:
        return jsonify({'message': 'Username or email already exists'}), 409
    

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = check_user_credentials(data['username'], data['password'])
    if user:
        access_token = create_access_token(identity=str(user.id))
        user_data = user.to_json()
        response = jsonify({
            'access_token': access_token,
            'user': user_data
        })
        response.set_cookie('access_token', access_token, httponly=True)
        return response, 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401


    
    