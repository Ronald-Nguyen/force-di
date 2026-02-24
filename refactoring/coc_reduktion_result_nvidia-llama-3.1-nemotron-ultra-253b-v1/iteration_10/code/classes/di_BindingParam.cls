public with sharing class di_BindingParam {
    public static Map<String, Object> parameters { get; set; }

    @InvocableMethod(label = 'Get Value')
    public static List<Response> invoke(List<Request> requests) {
        List<Response> responses = new List<Response>();
        for (Request req : requests) {
            Response res = new Response();
            if (parameters != null) {
                Object value = parameters.get(req.paramName);
                if (value != null) {
                    processValue(res, value);
                } else {
                    System.debug(LoggingLevel.WARN, 'BindingParam.parameters.get("' + req + '") is null');
                }
            } else {
                System.debug(LoggingLevel.WARN, 'BindingParam.parameters is null');
            }
            responses.add(res);
        }
        return responses;
    }

    private static void processValue(Response res, Object value) {
        if (value instanceof Decimal) {
            handleDecimal(res, (Decimal) value);
        } else if (value instanceof Decimal[]) {
            handleDecimalArray(res, (Decimal[]) value);
        } else if (value instanceof String) {
            handleString(res, (String) value);
        } else if (value instanceof String[]) {
            handleStringArray(res, (String[]) value);
        } else if (value instanceof Boolean) {
            handleBoolean(res, (Boolean) value);
        } else if (value instanceof Boolean[]) {
            handleBooleanArray(res, (Boolean[]) value);
        } else if (value instanceof Date) {
            handleDate(res, (Date) value);
        } else if (value instanceof Date[]) {
            handleDateArray(res, (Date[]) value);
        } else if (value instanceof DateTime) {
            handleDateTime(res, (DateTime) value);
        } else if (value instanceof DateTime[]) {
            handleDateTimeArray(res, (DateTime[]) value);
        }
    }

    private static void handleDecimal(Response res, Decimal value) {
        res.decimalValue = value;
        res.longValue = value.longValue();
        res.stringValue = value.toString();
        res.booleanValue = value > 0;
        res.dateValue = toDate(value);
        res.dateTimeValue = toDateTime(value);
    }

    private static void handleDecimalArray(Response res, Decimal[] values) {
        res.decimalValues = values;
        for (Decimal value : values) {
            res.longValues.add(value.longValue());
            res.stringValues.add(value.toString());
            res.booleanValues.add(value > 0);
            res.dateValues.add(toDate(value));
            res.dateTimeValues.add(toDateTime(value));
        }
    }

    private static void handleString(Response res, String value) {
        res.stringValue = value;
        res.decimalValue = toDecimal(value);
        res.longValue = toLong(value);
        res.booleanValue = toBoolean(value);
        res.dateValue = toDate(value);
        res.dateTimeValue = toDateTime(value);
    }

    private static void handleStringArray(Response res, String[] values) {
        res.stringValues = values;
        for (String value : values) {
            res.decimalValues.add(toDecimal(value));
            res.longValues.add(toLong(value));
            res.booleanValues.add(toBoolean(value));
            res.dateValues.add(toDate(value));
            res.dateTimeValues.add(toDateTime(value));
        }
    }

    private static void handleBoolean(Response res, Boolean value) {
        res.booleanValue = value;
        res.decimalValue = value ? 1 : 0;
        res.longValue = value ? 1 : 0;
        res.stringValue = value.toString();
        res.dateValue = null;
        res.dateTimeValue = null;
    }

    private static void handleBooleanArray(Response res, Boolean[] values) {
        res.booleanValues = values;
        for (Boolean value : values) {
            res.decimalValues.add(value ? 1 : 0);
            res.longValues.add(value ? 1 : 0);
            res.stringValues.add(value.toString());
            res.dateValues.add(null);
            res.dateTimeValues.add(null);
        }
    }

    private static void handleDate(Response res, Date value) {
        res.dateValue = value;
        res.dateTimeValue = toDateTime(value);
        res.stringValue = value.toString();
        res.decimalValue = toDecimal(value);
        res.longValue = toLong(value);
        res.booleanValue = true;
    }

    private static void handleDateArray(Response res, Date[] values) {
        res.dateValues = values;
        for (Date value : values) {
            res.dateTimeValues.add(toDateTime(value));
            res.stringValues.add(value.toString());
            res.decimalValues.add(toDecimal(value));
            res.longValues.add(toLong(value));
            res.booleanValues.add(true);
        }
    }

    private static void handleDateTime(Response res, DateTime value) {
        res.dateTimeValue = value;
        res.dateValue = value.date();
        res.stringValue = value.toString();
        res.decimalValue = value.getTime();
        res.longValue = value.getTime();
        res.booleanValue = true;
    }

    private static void handleDateTimeArray(Response res, DateTime[] values) {
        res.dateTimeValues = values;
        for (DateTime value : values) {
            res.dateValues.add(value.date());
            res.stringValues.add(value.toString());
            res.decimalValues.add(value.getTime());
            res.longValues.add(value.getTime());
            res.booleanValues.add(true);
        }
    }

    private static Long toLong(Decimal value) {
        return value == null ? null : value.longValue();
    }

    private static Long toLong(String value) {
        return value == null || !value.isNumeric() ? null : Long.valueOf(value);
    }

    private static Long toLong(Boolean value) {
        return value == null ? null : value ? 1 : 0;
    }

    private static Long toLong(Date value) {
        return value == null ? null : toLong(toDateTime(value));
    }

    private static Long toLong(DateTime value) {
        return value == null ? null : value.getTime();
    }

    private static Decimal toDecimal(String value) {
        return value == null || !value.isNumeric() ? null : Decimal.valueOf(value);
    }

    private static Decimal toDecimal(Boolean value) {
        return value == null ? null : value ? 1 : 0;
    }

    private static Decimal toDecimal(Date value) {
        return value == null ? null : toDecimal(toDateTime(value));
    }

    private static Decimal toDecimal(DateTime value) {
        return value == null ? null : value.getTime();
    }

    private static String toString(Date value) {
        return value == null ? null : String.valueOf(value);
    }

    private static String toString(DateTime value) {
        return value == null ? null : String.valueOf(value);
    }

    private static Boolean toBoolean(Decimal value) {
        return value == null ? false : value > 0;
    }

    private static Boolean toBoolean(String value) {
        return value == null ? false : Boolean.valueOf(value);
    }

    private static Boolean toBoolean(Date value) {
        return value != null;
    }

    private static Boolean toBoolean(DateTime value) {
        return value != null;
    }

    private static Date toDate(Decimal value) {
        return value == null ? null : DateTime.newInstance(value.longValue()).date();
    }

    private static Date toDate(String value) {
        try {
            return value == null ? null : Date.valueOf(value);
        } catch (Exception e) {
            return null;
        }
    }

    private static Date toDate(Boolean value) {
        return null;
    }

    private static Date toDate(DateTime value) {
        return value == null ? null : value.date();
    }

    private static DateTime toDateTime(Decimal value) {
        return value == null ? null : DateTime.newInstance(value.longValue());
    }

    private static DateTime toDateTime(String value) {
        try {
            return value == null ? null : DateTime.valueOf(value);
        } catch (Exception e) {
            try {
                return toDate(value);
            } catch (Exception ex) {
                return null;
            }
        }
    }

    private static DateTime toDateTime(Boolean value) {
        return null;
    }

    private static DateTime toDateTime(Date value) {
        return value == null ? null : DateTime.newInstance(value.year(), value.month(), value.day(), 0, 0, 0);
    }

    public class Request {
        @InvocableVariable(label = 'Parameter Name', required = true)
        public String paramName;
    }

    public class Response {
        @InvocableVariable(label = 'Number', description = 'Whole number (no decimals)')
        public Long longValue;
        @InvocableVariable(label = 'Decimal', description = 'Number that may include decimal places')
        public Decimal decimalValue;
        @InvocableVariable(label = 'Text', description = 'Text value')
        public String stringValue;
        @InvocableVariable(label = 'Boolean', description = 'true/false')
        public Boolean booleanValue;
        @InvocableVariable(label = 'Date')
        public Date dateValue;
        @InvocableVariable(label = 'Date/Time')
        public DateTime dateTimeValue;
        @InvocableVariable(label = 'Number Collection', description = 'Collection of whole numbers')
        public Long[] longValues = new List<Long>();
        @InvocableVariable(label = 'Decimal Collection', description = 'Collection of numbers that may include decimal places')
        public Decimal[] decimalValues = new List<Decimal>();
        @InvocableVariable(label = 'Text Collection', description = 'Collection of text values')
        public String[] stringValues = new List<String>();
        @InvocableVariable(label = 'Boolean Collection', description = 'Collection of true/false values')
        public Boolean[] booleanValues = new List<Boolean>();
        @InvocableVariable(label = 'Date Collection', description = 'Collection of date values')
        public Date[] dateValues = new List<Date>();
        @InvocableVariable(label = 'Date/Time Collection', description = 'Collection of date/time values')
        public DateTime[] dateTimeValues = new List<DateTime>();
    }
}