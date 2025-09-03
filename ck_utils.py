ck_str_keep_trading_hours = """(
                                (EXTRACT(HOUR FROM datetime) = 9 AND EXTRACT(MINUTE FROM datetime) >= 30)
                                OR (EXTRACT(HOUR FROM datetime) > 9 AND EXTRACT(HOUR FROM datetime) < 11)
                                OR (EXTRACT(HOUR FROM datetime) = 11 AND EXTRACT(MINUTE FROM datetime) <= 30)
                                OR (EXTRACT(HOUR FROM datetime) = 13)
                                OR (EXTRACT(HOUR FROM datetime) > 13 AND EXTRACT(HOUR FROM datetime) < 15)
                            )"""
