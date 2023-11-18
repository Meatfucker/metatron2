#settings.py This builds the settings variable and provides a function for getting server/channel defaults
#Used to be in metatron but the variable needs to be available in multiple functions and I dont feel like passing it around all the time

SETTINGS = {}

with open("settings.cfg", "r", encoding="utf-8") as settings_file:
    for line in settings_file:
        if "=" in line:
            key, value = (line.split("=", 1)[0].strip(), line.split("=", 1)[1].strip())
            if key in SETTINGS:
                if isinstance(SETTINGS[key], list):
                    SETTINGS[key].append(value)
                else:
                    SETTINGS[key] = [SETTINGS[key], value]
            else:
                SETTINGS[key] = [value]

async def get_defaults(idname):
    filename = f'defaults/{idname}.cfg'
    defaults = {}
    with open(filename, "r", encoding="utf-8") as defaults_file:
        for line in defaults_file:
            if "=" in line:
                key, value = (line.split("=", 1)[0].strip(), line.split("=", 1)[1].strip())
                if key in defaults:
                    if isinstance(defaults[key], list):
                        defaults[key].append(value)
                    else:
                        defaults[key] = [defaults[key], value]
                else:
                    defaults[key] = [value]
    return defaults