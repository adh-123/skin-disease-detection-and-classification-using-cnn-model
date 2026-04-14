#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os#integrate with os
import sys#handle command line arguments


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "skin_disease_project.settings")#use setting file from project
    try:
        from django.core.management import execute_from_command_line#thid importd mainDjango server comands
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)#this stores the command entered in the terminal


if __name__ == "__main__":
    main()
