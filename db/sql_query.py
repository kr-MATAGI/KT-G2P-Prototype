SELECT_ALL_ITEMS = """
    SELECT * FROM ?;
"""

CREATE_TABLE_SQL ={
    'Headword': """
        CREATE TABLE IF NOT EXISTS HeadWord(
            id integer PRIMARY KEY, 
            word TEXT not null, 
            pronun_list TEXT not null, 
            pos TEXT not null 
        ); 
    """,

    'ConjuWord': """
        CREATE TABLE IF NOT EXISTS ConjuWord(
            id integer PRIMARY KEY, 
            word TEXT not null, 
            pronun_list TEXT not null, 
            pos TEXT not null 
        );
    """,

    'NNPWord': """
        CREATE TABLE IF NOT EXISTS NNPWord(
            id integer PRIMARY KEY, 
            word TEXT not null, 
            pronun_list TEXT not null, 
            pos TEXT not null 
        );
    """
}

INSERT_ITEMS_SQL = {
    'HeadWord': """
        INSERT INTO HeadWord(word, pronun_list, pos) VALUES(?,?,?);
    """,

    'ConjuWord': """
        INSERT INTO ConjuWord(word, pronun_list, pos) VALUES(?,?,?);
    """,

    'NNPWord': """
        INSERT INTO NNPWord(word, pronun_list, pos) VALUES(?,?,?);
    """
}