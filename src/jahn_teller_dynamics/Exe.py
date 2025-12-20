#!/usr/bin/env python3


import sys
import traceback
import jahn_teller_dynamics.io.config.parser as new_jt_cfg
from jahn_teller_dynamics.io.workflow.orchestrator import JTOrchestrator

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
        # Create new config parser
        new_jt_config_parser = new_jt_cfg.JTConfigParser(config_file_name)
        
        # Create orchestrator and run calculation
        orchestrator = JTOrchestrator(new_jt_config_parser)
        print('Run an E⊗e dynamic Jahn-Teller calculation')
        orchestrator.run()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    

    


if __name__ == "__main__":
    main() 