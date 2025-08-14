#!/usr/bin/env python3


import sys
import jahn_teller_dynamics.io.JT_config_file_parsing as  JT_cfg    
import jahn_teller_dynamics.io.user_workflow as uw

def main():

    arguments = sys.argv[1:]
    
    if not arguments:
        print("Error: No configuration file specified.")
        print("Usage: Exe <config_file>")
        sys.exit(1)
    
    if len(arguments) == 1 and arguments[0] == '--version':
        try:
            from importlib.metadata import version
            print(f'jahn-teller-dynamics {version("jahn-teller-dynamics")}')
        except:
            print('jahn-teller-dynamics version unknown')
        sys.exit(0)


    
    config_file_name = arguments[0]
    
    try:
        JT_config_parser = JT_cfg.Jahn_Teller_config_parser(config_file_name)
        print('Run an Exe calculation')
        if JT_config_parser.is_ZPL_calculation():
            uw.ZPL_procedure(JT_config_parser)
        elif JT_config_parser.is_single_case():
            section_to_look_for = JT_cfg.single_case_section
            uw.spin_orbit_JT_procedure_general(JT_config_parser, section_to_look_for, complex_trf=True)
        else:
            print("Error: Could not determine calculation type from config file.")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_name}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 