from dotmap import DotMap

############ SETTINGS IMPORT ############

from .ai import ai
from .io import io
from .pepper import pepper
from .others import others

#########################################

demo_settings = DotMap()

demo_settings.ai = ai
demo_settings.io = io
demo_settings.pepper = pepper
demo_settings.others = others