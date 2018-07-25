import logging

import click
import yaml

import data
import model

@click.group()
@click.option('-p', '--params-path', default='parameters.yml')
@click.pass_context
def main(ctx, params_path):
    ctx.obj = {'params_path': params_path}


@main.command('train')
@click.pass_context
def train(ctx):
    raise NotImplementedError

@main.command('test')
def test(ctx):
    raise NotImplementedError

@main.command('infer')
def infer(ctx):
    raise NotImplementedError

@main.command('download-data')
def download_data(ctx):
    raise NotImplementedError

@main.command('preprocess-data')
def preprocess_data(ctx):
    raise NotImplementedError

    
if __name__ == '__name__':
    main(obj={})