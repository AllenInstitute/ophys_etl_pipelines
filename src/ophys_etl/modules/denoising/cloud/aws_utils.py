from typing import Optional

import boto3


def get_account_id(boto_session: Optional[boto3.Session] = None,
                   profile_name: str = 'default',
                   region_name: str = 'us-west-2') -> str:
    """
    Gets the AWS account id

    Parameters
    ----------
    boto_session
        Optional boto session. If not provided, will create one
    profile_name
        AWS profile name
    region_name
        AWS region

    Returns
    -------
    AWS Account id
    """
    if boto_session is None:
        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
    sts = boto_session.client('sts')
    id = sts.get_caller_identity()
    return id['Account']
