"""Functions for building docker and uploading to AWS Elastic Container Service
(ECR)"""

import base64
from pathlib import Path

import boto3
import docker
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_and_push_container(repository_name: str, dockerfile_dir: Path,
                             image_tag: str, profile_name='default'):
    session = boto3.session.Session(profile_name=profile_name,
                                    region_name='us-west-2')
    account = _get_account(session=session)
    region = session.region_name
    image_name = repository_name
    repository_host = f'{account}.dkr.ecr.{region}.amazonaws.com'

    logger.info('Creating ecr repository')
    _create_repository(session=session, name=repository_name)

    logger.info('Getting ecr credentials')
    docker_client, ecr_username, ecr_password = _docker_login(
        session=session,
        registry_name=repository_host)

    logger.info('Building docker image')
    _build_docker(docker_client=docker_client,
                  path_to_dockerfile=dockerfile_dir,
                  image_name=image_name, image_tag=image_tag)

    logger.info('Tagging locally')
    _tag_docker(docker_client=docker_client, image_name=image_name,
                image_tag=image_tag, repository_name=repository_name,
                repository_host=repository_host)

    logger.info('Pushing image to ecr')
    _docker_push(docker_client=docker_client, repository_host=repository_host,
                 image_name=image_name,
                 image_tag=image_tag, username=ecr_username,
                 password=ecr_password)


def _get_account(session):
    sts = session.client('sts')
    id = sts.get_caller_identity()
    return id['Account']


def _create_repository(session, name: str, region='us-west-2'):
    ecr = session.client('ecr', region_name=region)
    repos = ecr.describe_repositories()
    repos = repos['repositories']
    if name in [x['repositoryName'] for x in repos]:
        logger.info(f'ECR repository {name} already exists')
        return
    ecr.create_repository(repositoryName=name)


def _docker_login(session, registry_name: str, region='us-west-2'):
    ecr = session.client('ecr', region_name=region)
    auth = ecr.get_authorization_token()
    auth = auth['authorizationData'][0]
    password = base64.b64decode(auth['authorizationToken'])
    password = bytes.decode(password)
    password = password.replace('AWS:', '')
    client = docker.APIClient()
    client.login(username='AWS', password=password, registry=registry_name)
    return client, 'AWS', password


def _build_docker(docker_client: docker.APIClient, path_to_dockerfile: Path,
                  image_name: str, image_tag: str):
    image_tag = f'{image_name}:{image_tag}'
    build_res = docker_client.build(path=str(path_to_dockerfile),
                                    tag=image_tag, rm=True, decode=True)
    for line in build_res:
        logger.info(line)
        if 'errorDetail' in line:
            raise RuntimeError(line['errorDetail']['message'])


def _tag_docker(docker_client: docker.APIClient, image_name: str,
                repository_name: str, image_tag: str, repository_host: str):
    full_repository_name = f'{repository_host}/{repository_name}'
    tag_successful = docker_client.tag(image=image_name,
                                       repository=full_repository_name,
                                       tag=image_tag)
    if not tag_successful:
        raise RuntimeError('Tagging was not successful')


def _docker_push(docker_client: docker.APIClient, repository_host: str,
                 image_name: str,
                 image_tag: str, username: str, password: str):
    repository = f'{repository_host}/{image_name}'
    auth_config = {
        'username': username,
        'password': password
    }
    res = docker_client.push(repository=repository, tag=image_tag,
                             auth_config=auth_config, decode=True, stream=True)
    for line in res:
        logger.info(line)
        if 'errorDetail' in line:
            raise RuntimeError(res['errorDetail']['message'])

