from werkzeug.security import generate_password_hash, check_password_hash
from ..schemas.user_schema import User
from mongoengine.errors import NotUniqueError
import logging

def register_user(username, email, password, role='client', organization='', contact=''):
    try:
        hashed_password = generate_password_hash(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
            organization=organization,
            contact=contact             
        )
        user.save()
        return True
    except NotUniqueError:
        return False 
    

def check_user_credentials(username, password):
    try:
        user = User.objects(username=username).first()
        if user and check_password_hash(user.hashed_password, password):
            return user  # Authentication successful
        else:
            return None  # Authentication failed
    except Exception as e:
        logging.error(f"An error occurred during login: {e}")
        return None

