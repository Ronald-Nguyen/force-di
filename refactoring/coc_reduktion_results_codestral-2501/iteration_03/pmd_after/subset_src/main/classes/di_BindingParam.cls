public with sharing class di_BindingParam {

    public static Map<String, Object> parameters { get; set; }

    @InvocableMethod(
        label = 'Get Value'
    )
    public static List<Response> invoke(List<Request> requests) {
        List<Response> responses = new List<Response>();

        for (Request req : requests) {
            Response res = processRequest(req);
            responses.add(res);
        }

        return responses;
    }

    private static Response processRequest(Request req) {
        Response res = new Response();

        if (parameters == null) {
            System.debug(LoggingLevel.WARN, 'BindingParam.parameters is null');
            return res;
        }

        Object value = parameters.get(req.paramName);
        if (value == null) {
            System.debug(LoggingLevel.WARN, 'BindingParam.parameters.get("' + req + '") is null');
            return res;
        }

        processValue(value, res);
        ensureSingleValues(res);
        ensureCollectionValues(res);

        System.debug('paramName=' + req);
        System.debug(res);

        return res;
    }

    private static void processValue(Object value, Response res) {
        if (value instanceof Decimal) {
            processDecimalValue((Decimal)value, res);
        } else if (value instanceof Decimal[]) {
            processDecimalArray((Decimal[])value, res);
        } else if (value instanceof String) {
            processStringValue((String)value, res);
        } else if (value instanceof String[]) {
            processStringArray((String[])value, res);
        } else if (value instanceof Boolean) {
            processBooleanValue((Boolean)value, res);
        } else if (value instanceof Boolean[]) {
            processBooleanArray((Boolean[])value, res);
        } else if (value instanceof Date) {
            processDateValue((Date)value, res);
        } else if (value instanceof Date[]) {
            processDateArray((Date[])value, res);
        } else if (value instanceof DateTime) {
            processDateTimeValue((DateTime)value, res);
        } else if (value instanceof DateTime[]) {
            processDateTimeArray((DateTime[])value, res);
        }
    }

    private static void processDecimalValue(Decimal value, Response res) {
        res.longValue = toLongValue(value);
        res.decimalValue = value;
        res.stringValue = toStringValue(value);
        res.booleanValue = toBooleanValue(value);
        res.dateValue = toDateValue(value);
        res.dateTimeValue = toDateTimeValue(value);
    }

    private static void processDecimalArray(Decimal[] values, Response res) {
        res.decimalValues = values;
        for (Decimal value : values) {
            res.longValues.add(toLongValue(value));
            res.stringValues.add(toStringValue(value));
            res.booleanValues.add(toBooleanValue(value));
            res.dateValues.add(toDateValue(value));
            res.dateTimeValues.add(toDateTimeValue(value));
        }
    }

    private static void processStringValue(String value, Response res) {
        res.longValue = toLongValue(value);
        res.decimalValue = toDecimalValue(value);
        res.stringValue = value;
        res.booleanValue = toBooleanValue(value);
        res.dateValue = toDateValue(value);
        res.dateTimeValue = toDateTimeValue(value);
    }

    private static void processStringArray(String[] values, Response res) {
        res.stringValues = values;
        for (String value : values) {
            res.longValues.add(toLongValue(value));
            res.decimalValues.add(toDecimalValue(value));
            res.booleanValues.add(toBooleanValue(value));
            res.dateValues.add(toDateValue(value));
            res.dateTimeValues.add(toDateTimeValue(value));
        }
    }

    private static void processBooleanValue(Boolean value, Response res) {
        res.longValue = toLongValue(value);
        res.decimalValue = toDecimalValue(value);
        res.stringValue = toStringValue(value);
        res.booleanValue = value;
        res.dateValue = toDateValue(value);
        res.dateTimeValue = toDateTimeValue(value);
    }

    private static void processBooleanArray(Boolean[] values, Response res) {
        res.booleanValues = values;
        for (Boolean value : values) {
            res.longValues.add(toLongValue(value));
            res.decimalValues.add(toDecimalValue(value));
            res.stringValues.add(toStringValue(value));
            res.dateValues.add(toDateValue(value));
            res.dateTimeValues.add(toDateTimeValue(value));
        }
    }

    private static void processDateValue(Date value, Response res) {
        res.longValue = toLongValue(value);
        res.decimalValue = toDecimalValue(value);
        res.stringValue = toStringValue(value);
        res.booleanValue = toBooleanValue(value);
        res.dateValue = value;
        res.dateTimeValue = toDateTimeValue(value);
    }

    private static void processDateArray(Date[] values, Response res) {
        res.dateValues = values;
        for (Date value : values) {
            res.longValues.add(toLongValue(value));
            res.decimalValues.add(toDecimalValue(value));
            res.stringValues.add(toStringValue(value));
            res.booleanValues.add(toBooleanValue(value));
            res.dateTimeValues.add(toDateTimeValue(value));
        }
    }

    private static void processDateTimeValue(DateTime value, Response res) {
        res.longValue = toLongValue(value);
        res.decimalValue = toDecimalValue(value);
        res.stringValue = toStringValue(value);
        res.booleanValue = toBooleanValue(value);
        res.dateValue = toDateValue(value);
        res.dateTimeValue = value;
    }

    private static void processDateTimeArray(DateTime[] values, Response res) {
        res.dateTimeValues = values;
        for (DateTime value : values) {
            res.longValues.add(toLongValue(value));
            res.decimalValues.add(toDecimalValue(value));
            res.stringValues.add(toStringValue(value));
            res.booleanValues.add(toBooleanValue(value));
            res.dateValues.add(toDateValue(value));
        }
    }

    private static void ensureSingleValues(Response res) {
        if (res.longValue == null && !res.longValues.isEmpty()) {
            res.longValue = res.longValues[0];
        }
        if (res.decimalValue == null && !res.decimalValues.isEmpty()) {
            res.decimalValue = res.decimalValues[0];
        }
        if (res.stringValue == null && !res.stringValues.isEmpty()) {
            res.stringValue = res.stringValues[0];
        }
        if (res.booleanValue == null && !res.booleanValues.isEmpty()) {
            res.booleanValue = res.booleanValues[0];
        }
        if (res.dateValue == null && !res.dateValues.isEmpty()) {
            res.dateValue = res.dateValues[0];
        }
        if (res.dateTimeValue == null && !res.dateTimeValues.isEmpty()) {
            res.dateTimeValue = res.dateTimeValues[0];
        }
    }

    private static void ensureCollectionValues(Response res) {
        if (res.longValue != null && res.longValues.isEmpty()) {
            res.longValues.add(res.longValue);
        }
        if (res.decimalValue != null && res.decimalValues.isEmpty()) {
            res.decimalValues.add(res.decimalValue);
        }
        if (res.stringValue != null && res.stringValues.isEmpty()) {
            res.stringValues.add(res.stringValue);
        }
        if (res.booleanValue != null && res.booleanValues.isEmpty()) {
            res.booleanValues.add(res.booleanValue);
        }
        if (res.dateValue != null && res.dateValues.isEmpty()) {
            res.dateValues.add(res.dateValue);
        }
        if (res.dateTimeValue != null && res.dateTimeValues.isEmpty()) {
            res.dateTimeValues.add(res.dateTimeValue);
        }
    }

    private static Long toLongValue(Decimal value) {
        return (value == null ? null : value.longValue());
    }

    private static Long toLongValue(String value) {
        return ((value == null || !value.isNumeric()) ? null : Long.valueOf(value));
    }

    private static Long toLongValue(Boolean value) {
        return (value == null ? null : value ? 1 : 0);
    }

    private static Long toLongValue(Date value) {
        return (value == null ? null : toLongValue(
            DateTime.newInstance(value.year(), value.month(), value.day(), 0, 0, 0)
        ));
    }

    private static Long toLongValue(DateTime value) {
        return (value == null ? null : value.getTime());
    }

    private static Decimal toDecimalValue(String value) {
        return ((value == null || !value.isNumeric()) ? null : Decimal.valueOf(value));
    }

    private static Decimal toDecimalValue(Boolean value) {
        return (value == null ? null : value ? 1 : 0);
    }

    private static Decimal toDecimalValue(Date value) {
        return (value == null ? null : toDecimalValue(
            DateTime.newInstance(value.year(), value.month(), value.day(), 0, 0, 0)
        ));
    }

    private static Decimal toDecimalValue(DateTime value) {
        return (value == null ? null : value.getTime());
    }

    private static String toStringValue(Date value) {
        return (value == null ? null : String.valueOf(value));
    }

    private static String toStringValue(DateTime value) {
        return (value == null ? null : String.valueOf(value));
    }

    private static String toStringValue(Object value) {
        return (value == null ? null : String.valueOf(value));
    }

    private static Boolean toBooleanValue(Decimal value) {
        return (value == null ? false : (value > 0));
    }

    private static Boolean toBooleanValue(String value) {
        return (value == null ? false : Boolean.valueOf(value));
    }

    private static Boolean toBooleanValue(Date value) {
        return (value == null ? false : true);
    }

    private static Boolean toBooleanValue(DateTime value) {
        return (value == null ? false : true);
    }

    private static Date toDateValue(Decimal value) {
        return (value == null ? null : DateTime.newInstance(value.longValue()).date());
    }

    private static Date toDateValue(String value) {
        try {
            return (value == null ? null : Date.valueOf(value));
        } catch (Exception e) {
            return null;
        }
    }

    private static Date toDateValue(Boolean value) {
        return null;
    }

    private static Date toDateValue(DateTime value) {
        return (value == null ? null : value.date());
    }

    private static DateTime toDateTimeValue(Decimal value) {
        return (value == null ? null : DateTime.newInstance(value.longValue()));
    }

    private static DateTime toDateTimeValue(String value) {
        try {
            return (value == null ? null : DateTime.valueOf(value));
        } catch (Exception e) {
            return toDateValue(value);
        }
    }

    private static DateTime toDateTimeValue(Boolean value) {
        return null;
    }

    private static DateTime toDateTimeValue(Date value) {
        return (value == null ? null : DateTime.newInstance(value.year(), value.month(), value.day(), 0, 0, 0));
    }

    public class Request {
        @InvocableVariable(
            label = 'Parameter Name',
            required = true
        )
        public String paramName;
    }

    public class Response {
        @InvocableVariable(
            label = 'Number',
            description = 'Whole number (no decimals)'
        )
        public Long longValue;

        @InvocableVariable(
            label = 'Decimal',
            description = 'Number that may include decimal places'
        )
        public Decimal decimalValue;

        @InvocableVariable(
            label = 'Text',
            description = 'Text value'
        )
        public String stringValue;

        @InvocableVariable(
            label = 'Boolean',
            description = 'true/false'
        )
        public Boolean booleanValue;

        @InvocableVariable(
            label = 'Date'
        )
        public Date dateValue;

        @InvocableVariable(
            label = 'Date/Time'
        )
        public DateTime dateTimeValue;

        @InvocableVariable(
            label = 'Number Collection',
            description = 'Collection of whole numbers'
        )
        public Long[] longValues = new Long[] {};

        @InvocableVariable(
            label = 'Decimal Collection',
            description = 'Collection of numbers that may include decimal places'
        )
        public Decimal[] decimalValues = new Decimal[] {};

        @InvocableVariable(
            label = 'Text Collection',
            description = 'Collection of text values'
        )
        public String[] stringValues = new String[] {};

        @InvocableVariable(
            label = 'Boolean Collection',
            description = 'Collection of true/false values'
        )
        public Boolean[] booleanValues = new Boolean[] {};

        @InvocableVariable(
            label = 'Date Collection',
            description = 'Collection of date values'
        )
        public Date[] dateValues = new Date[] {};

        @InvocableVariable(
            label = 'Date/Time Collection',
            description = 'Collection of date/time values'
        )
        public DateTime[] dateTimeValues = new DateTime[] {};
    }
}