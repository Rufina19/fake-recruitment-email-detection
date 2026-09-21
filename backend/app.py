from flask import Flask
from flask_cors import CORS
from routes import api_bp, ui_bp
from database import init_db


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    CORS(app)

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(ui_bp)

    init_db()
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

