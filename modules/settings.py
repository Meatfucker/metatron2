# settings.py This builds the settings variable and provides a function for getting server/channel defaults

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
    try:
        with open(filename, "r", encoding="utf-8") as defaults_file:
            for defaults_line in defaults_file:
                if "=" in defaults_line:
                    defaults_key, defaults_value = (defaults_line.split("=", 1)[0].strip(), defaults_line.split("=", 1)[1].strip())
                    if defaults_key in defaults:
                        if isinstance(defaults[defaults_key], list):
                            defaults[defaults_key].append(defaults_value)
                        else:
                            defaults[defaults_key] = [defaults[defaults_key], defaults_value]
                    else:
                        defaults[defaults_key] = [defaults_value]
    except FileNotFoundError:
            return None
    return defaults
