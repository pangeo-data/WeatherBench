configfile: 'scripts/config_default.yml'
all_years = [
    '1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993',
    '1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008',
    '2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'
]

if config['tmpdir'] == 'None':
    config['tmpdir'] = config['datadir']

print(config)

rule download:
    output:
          "{dataset}/raw/{name}/{name}_{year}_raw.nc",
    shell:
         "python src/download.py single \
            --variable {config[era_var_name]} \
            --level_type {config[level_type]} \
            --pressure_level {config[pressure_level]} \
            --output_dir {wildcards.dataset}/raw/{wildcards.name} \
            --custom_fn {wildcards.name}_{wildcards.year}_raw.nc \
            --years {wildcards.year}"

rule regrid:
    input:
          "{dataset}/raw/{name}/{name}_{year}_raw.nc"
    output:
          "{dataset}/{res}deg/{name}/{name}_{year}_{res}deg.nc.tmp"
    priority: 100
    threads: 3
    shell:
          "python src/regrid.py \
            --input_fns {input} \
            --output_dir {wildcards.dataset}/{wildcards.res}deg/{wildcards.name} \
            --ddeg_out {wildcards.res} \
            --file_ending nc.tmp"

rule delete:
    input:
          expand("{tmpdir}/{res}deg/{{name}}/{{name}}_{{year}}_{res}deg.nc.tmp",
                 res=config['res'], tmpdir=config['tmpdir']),
    output:
          expand("{{dataset}}/{res}deg/{{name}}/{{name}}_{{year}}_{res}deg.nc",
                 res=config['res'])
    priority: 100
    run:
          for i, o in zip(input, output):
              shell("mv {i} {o}")
          shell("rm {config[tmpdir]}/raw/{wildcards.name}/{wildcards.name}_{wildcards.year}_raw.nc"),


rule zip:
    input:
         expand("{{dataset}}/{{res}}deg/{{name}}/{{name}}_{year}_{{res}}deg.nc",
                year=all_years)
    output:
          "{dataset}/{res}deg/{name}/{name}_{res}deg.zip"
    shell:
         "cd {wildcards.dataset}/{wildcards.res}deg/{wildcards.name}/ && zip {output} *.nc && "
         "rm {wildcards.dataset}/{wildcards.res}deg/{wildcards.name}/*.nc"

rule all_zip:
    input:
         expand("{datadir}/{res}deg/{name}/{name}_{res}deg.zip",
                datadir=config['datadir'], res=config['res'], name=config['name']),

rule all_nozip:
    input:
         expand("{datadir}/{res}deg/{name}/{name}_{year}_{res}deg.nc",
                datadir=config['datadir'], res=config['res'], name=config['name'], year=all_years),

