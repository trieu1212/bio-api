from flask import Flask, jsonify
from api.routes import user_bp, auth_bp

app = Flask(__name__) 

app.register_blueprint(user_bp, prefix='/api/user')
app.register_blueprint(auth_bp, prefix='/api/auth')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test endpoint is working!'}), 200

if __name__ == '__main__':
    app.run(debug=True)

