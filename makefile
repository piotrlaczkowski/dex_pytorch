format:
	echo "Formating Pyton Code"
	pre-commit run --all

update-hooks:
	pre-commit autoupdate

prepare-deploy:
	pip install torch-model-archiver
	torch-model-archiver --model-name Madnes --serialized-file models/model1.pt --extra-files models/preprocessor.pkl --version 1.0 --requirements-file requirements.txt
