CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS Dict(
    id integer PRIMARY KEY, 
    word TEXT not null, 
    pronun_list TEXT not null, 
    pos TEXT not null 
    );
"""

INSERT_SQL = """
    INSERT INTO Dict(word, pronun_list, pos) VALUES(?,?,?);
"""

SELECT_ALL_ITEMS = """
    SELECT * FROM Dict;
"""