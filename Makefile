.ONESHELL:
SHELL := /bin/zsh

init-db: setup-db setup-users setup-interviews
	@echo "Database initialized with admins and interview guide"

setup-db:
		python -m app.db --recreate-db

setup-users:
		python -m app.db --create-users

setup-interviews:
	python -m app.db --create-project \
		--interview-title "Øresundskollegiet - Images" \
		--interview-guide-path data/interview_guides/ro_with_images.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Øresundskollegiet" \
		--interview-guide-path data/interview_guides/ro_without_images.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Deservingness - A" \
		--interview-guide-path data/interview_guides/deservingness_a.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Deservingness - B" \
		--interview-guide-path data/interview_guides/deservingness_b.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Deservingness - C" \
		--interview-guide-path data/interview_guides/deservingness_c.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Survey test" \
		--interview-guide-path data/interview_guides/background_survey.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "volunteer - long" \
		--interview-guide-path data/interview_guides/volunteer_long.json \
		--interview-config-path data/configs/deservingness_config.yaml
	python -m app.db --create-project \
		--interview-title "Image test" \
		--interview-guide-path data/interview_guides/with_images.json \
		--interview-config-path data/configs/deservingness_config.yaml

cache:
	redis-server redis.conf

message ?=
migrate-db:
	@if [ -z "$(message)" ]; then \
		echo "Error: message is not set. Usage: make migrate-db message=value"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(message)"
	alembic upgrade head

set-proxy:
	bash scripts/get_ip.sh > .proxy_host

db-name=ainterviewer
path-to-db=./app
db-ext=sqlite
output-directory=./docs/diagrams
output-file=db_diagram.png
db-diagram:
	mkdir -p /tmp/schemaspy; \
	schemaspy -dp /usr/share/java/sqlite-jdbc/ \
		-t sqlite-xerial \
		-cat ${db-name} -s ${db-name} -db ${path-to-db}/${db-name}.${db-ext} \
		-sso -o /tmp/schemaspy/ > /tmp/schemaspy/diagram.log 2>&1 && \
		mkdir --parents $(output-directory); mv /tmp/schemaspy/diagrams/summary/relationships.real.large.png ${output-directory}/${output-file}

db-show:
	sqlitebrowser app/ainterviewer.sqlite > /tmp/null 2>&1 & disown

get-utils:
	wget -O models/convert_hf_to_gguf.py https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py

# Project setup
setup: setup-base setup-db
	@printf "\nAn enviroment has been created:\n\tainterviewer\n\n\
		Activate it and install dependencies from the pyproject.toml file, ie. `pip install -e '.[all]'`\n\n\
		A database has been created into $DATABASE_FILE"

setup-base:
	touch .env
	echo "[]" >> data/users.json

n-agents=10
lan=en
synthesize:
	python -m ainterviewer.synthesize.interviews \
		--num-agents $(n-agents) \
		--lan $(lan) \
		--answering-model $(model) \
		--interview-model-config $(config)
