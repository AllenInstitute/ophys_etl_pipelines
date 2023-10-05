import logging
import os
import subprocess

import argschema
import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class DeploySchema(argschema.ArgSchema):
    slack_token = argschema.fields.String(
        metadata={
            'description': (
                'slack token. See https://api.slack.com/messaging/sending')
        },
        required=True
    )

    channel_id = argschema.fields.String(
        metadata={
            'description': 'Channel to send message to.'
        },
        dump_default='C05VAB63YJ1'
    )

    values_yaml_path = argschema.fields.InputFile(
        metadata={
            'descripton': 'Path to values.yaml file for helm'
        },
        required=True
    )

    ophys_etl_pipelines_root_dir = argschema.fields.InputDir(
        metadata={
            'description': 'Path to ophys_etl_pipelines root dir'
        },
        required=True
    )


class DeployRunner(argschema.ArgSchemaParser):
    """Deploys app using helm and notifies slack channel"""

    default_schema = DeploySchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger.setLevel(level=logging.INFO)

    def run(self):
        self._deploy_app()
        self.logger.info('Deployment successful')

        self._notify_slack_channel()
        self.logger.info('Slack channel notified')

    def _deploy_app(self):
        """Runs helm upgrade command to deploy app"""
        cmd = [
            'helm',
            'upgrade',
            f'{os.environ["HELM_RELEASE_NAME"]}',
            'apache-airflow/airflow',
            '--namespace',
            f'{os.environ["KUBERNETES_NAMESPACE"]}',
            '-f',
            f'{self.args["values_yaml_path"]}'
        ]
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f'Could not deploy app: '
                               f'{result.stderr}')
        output = result.stdout
        self.logger.info(output)

    def _notify_slack_channel(self):
        """Sends slack channel a message"""
        tag = self._get_tag_to_deploy()
        current_commit = self._get_current_commit()
        new_commit = self._get_new_commit()

        text = ('Ophys processing was deployed!\n'
                f'*Tag*: {tag}\n'
                f'https://github.com/AllenInstitute/ophys_etl_pipelines/compare/{current_commit}...{new_commit}')    # noqa E402

        client = WebClient(token=self.args['slack_token'])

        try:
            client.chat_postMessage(
                channel=self.args['channel_id'],
                text=text)
        except SlackApiError as e:
            self.logger.error(f'Error: {e}')

    def _get_tag_to_deploy(self):
        """Gets tag from values.yaml file"""
        with open(self.args['values_yaml_path']) as f:
            values = yaml.safe_load(f)
        tag = values['images']['airflow']['tag']
        return tag

    def _get_new_commit(self):
        """Gets the commit sha to be deployed"""
        cmd = [
            'git',
            '-C',
            f'{self.args["ophys_etl_pipelines_root_dir"]}',
            'rev-parse',
            'HEAD'
        ]
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f'Could not get git commit sha to be deployed: '
                               f'{result.stderr}')
        output = result.stdout
        return output

    @staticmethod
    def _get_current_commit():
        """Gets the currently deployed git commit sha"""
        cmd = [
            'kubectl',
            'exec',
            '--namespace',
            f'{os.environ["KUBERNETES_NAMESPACE"]}',
            'ophys-processing-scheduler-0',
            '--',
            'bash',
            '-c',
            'cd ophys_etl_pipelines && git rev-parse HEAD'
        ]
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f'Could not get currently deployed git commit '
                               f'sha: {result.stderr}')
        # returns something like
        # Defaulted container "scheduler" out of: scheduler, scheduler-log-groomer, wait-for-airflow-migrations (init)  # noqa E402
        # 8a6883bb7924869b5480f6e9ad6d61ad73298015
        # get the sha
        output = result.stdout.splitlines()[-1]

        return output


if __name__ == '__main__':
    runner = DeployRunner()
    runner.run()
