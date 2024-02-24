import utilities.OUTCAR_parsing as out_pars


filename = "C3v.outcar"
parser = out_pars.OUTCAR_data_parser(filename)

print(parser.energy)
print('fin')