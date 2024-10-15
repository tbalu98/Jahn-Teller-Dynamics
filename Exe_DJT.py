import utilities.user_workflow as uw
import sys
import utilities.JT_config_file_parsing as  JT_cfg

arguments = sys.argv[1:]


config_file_name = arguments[0]


JT_config_parser = JT_cfg.Jahn_Teller_config_parser(config_file_name)

if JT_config_parser.is_ZPL_calculation():
    
    uw.ZPL_procedure(JT_config_parser)

elif JT_config_parser.is_single_case():


    section_to_look_for = JT_cfg.single_case_section

    uw.spin_orbit_JT_procedure_general(JT_config_parser, section_to_look_for, complex_trf=True)