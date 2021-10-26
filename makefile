format:
	echo "Formating Pyton Code"
	pre-commit run --all

update-hooks:
	pre-commit autoupdate
