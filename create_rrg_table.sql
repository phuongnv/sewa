-- Create table to store RRG (Relative Strength Group) time series data
CREATE TABLE IF NOT EXISTS rrg_data (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    rs_ratio_scaled NUMERIC(10, 4) NOT NULL,
    rs_momentum_scaled NUMERIC(10, 4) NOT NULL,
    close NUMERIC(12, 2),
    rs_fa NUMERIC(20, 17),
    rm_fa NUMERIC(20, 17),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- Create index for faster queries by symbol
CREATE INDEX IF NOT EXISTS idx_rrg_data_symbol ON rrg_data(symbol);

-- Create index for faster queries by date
CREATE INDEX IF NOT EXISTS idx_rrg_data_date ON rrg_data(date);

-- Create index for faster queries by symbol and date range
CREATE INDEX IF NOT EXISTS idx_rrg_data_symbol_date ON rrg_data(symbol, date);

COMMENT ON TABLE rrg_data IS 'Stores RRG (Relative Strength Group) time series data with RS-Ratio and RS-Momentum scaled values';
COMMENT ON COLUMN rrg_data.symbol IS 'Stock symbol';
COMMENT ON COLUMN rrg_data.date IS 'Trading date';
COMMENT ON COLUMN rrg_data.rs_ratio_scaled IS 'RS-Ratio scaled value (centered at 100)';
COMMENT ON COLUMN rrg_data.rs_momentum_scaled IS 'RS-Momentum scaled value (centered at 100)';
COMMENT ON COLUMN rrg_data.close IS 'Closing price for the date';
COMMENT ON COLUMN rrg_data.rs_fa IS 'RS value from FireAnt RRG API';
COMMENT ON COLUMN rrg_data.rm_fa IS 'RM value from FireAnt RRG API';

